from llm_transparency_tool.models.transparent_llm import TransparentLlm, ModelInfo
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float
import torch
from torch import Tensor
from typing import cast
class LayerSkipLlama(TransparentLlm):
    def __init__(self, model_path="facebook/layerskip-llama2-7B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
            output_attentions=True,
            use_auth_token=True
        )

        self._last_batch = None
        self._logits = None
        self._hidden_states = None
        self._attentions = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # For implementation
        self._residual_after_attn = {}

        for i, block in enumerate(self.model.model.layers):
            block.self_attn.register_forward_hook(self._make_hook(i))


        self._V_per_layer = {}
        for i, block in enumerate(self.model.model.layers):
            block.self_attn.v_proj.register_forward_hook(self._make_v_hook(i))

    def model_info(self) -> ModelInfo:
        config = self.model.config
        return ModelInfo(
            name=config._name_or_path,
            n_params_estimate=int(self.model.num_parameters() / 1e6),
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            d_model=config.hidden_size,
            d_vocab=config.vocab_size,
        )

    def run(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model(**tokens)
        self._last_batch = tokens
        self._logits = output.logits
        self._hidden_states = output.hidden_states
        self._attentions = output.attentions

    def copy(self):
        return LayerSkipLlama(model_path=self.model.name_or_path)

    def batch_size(self):
        return self._last_batch["input_ids"].shape[0]

    def tokens(self):
        return self._last_batch["input_ids"]

    def tokens_to_strings(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def logits(self):
        return self._logits

    def unembed(self, t, normalize):
        if normalize:
            t = torch.nn.functional.layer_norm(t, t.shape[-1:])
        return torch.matmul(t, self.model.lm_head.weight.T)

    def residual_in(self, layer):
        return self._hidden_states[layer]

    def residual_out(self, layer):
        return self._hidden_states[layer + 1]

    def attention_matrix(self, batch_i, layer, head):
        return self._attentions[layer][batch_i, head]

    # Implementations
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]  # first element is hidden_states
            else:
                attn_output = output
            self._residual_after_attn[layer_idx] = attn_output.detach()

        return hook

    # Leave the rest as NotImplemented
    def residual_after_attn(self, layer):
        return self._residual_after_attn[layer]

    def ffn_out(self, layer):
        # Just reuse the post-layer hidden state for now
        return self._hidden_states[layer + 1]

    def decomposed_ffn_out(self, batch_i, layer, pos):
        # Return a placeholder list of tensors
        dim = self._hidden_states[layer + 1].shape[-1]
        return [torch.zeros(dim).to(self.device) for _ in range(10)]

    def neuron_activations(self, batch_i, layer, pos):
        # Fake 10 neuron activations
        return torch.zeros(10).to(self.device)

    def neuron_output(self, layer, neuron):
        # Return a vector with 1 in the neuron position
        dim = self._hidden_states[layer + 1].shape[-1]
        vec = torch.zeros(dim).to(self.device)
        vec[neuron] = 1.0
        return vec

    def attention_output(self, batch_i, layer, pos, head):
        # Grab the attention weights for this layer: shape [batch, num_heads, q_len, k_len]
        attn = self._attentions[layer][batch_i, head, pos]  # [key_len]

        # Use the values from the input tokens (pre-projection): shape [batch, seq_len, hidden]
        value_vectors = self._hidden_states[layer][batch_i]  # [seq_len, hidden_dim]

        # Compute weighted sum of values using attention weights
        attn_output = torch.matmul(attn, value_vectors)  # [hidden_dim]
        return attn_output.detach()



    def _register_value_hooks(self):
        for i, block in enumerate(self.model.model.layers):
            def value_hook(module, input, output):
                # output is a tuple: (hidden_states, attn_weights)
                # You need to extract V manually — might require modifying the model
                # or manually redoing the forward pass with saved weights
                ...

            block.self_attn.register_forward_hook(value_hook)

    def _make_v_hook(self, layer_idx):
        def hook(module, input, output):
            # Save V for later use
            self._V_per_layer[layer_idx] = output.detach()

        return hook

    def decomposed_attn(self, batch_i, layer) -> Float[Tensor, "batch pos key_pos head d_model"]:
        if layer not in self._V_per_layer:
            print(f"[WARN] V not captured for layer {layer}, returning zeros")
            seq_len = self._hidden_states[layer + 1].shape[1]
            hidden_dim = self._hidden_states[layer + 1].shape[2]
            num_heads = self.model.config.num_attention_heads

            zeros = torch.zeros(1, seq_len, seq_len, num_heads, hidden_dim, device=self.device)
            return cast(Float[Tensor, "batch pos key_pos head d_model"], zeros)

        attn = self._attentions[layer][batch_i]  # [num_heads, pos, key_pos]
        V = self._V_per_layer[layer][batch_i]  # [seq_len, hidden_dim]

        num_heads = attn.shape[0]
        d_model = V.shape[-1]
        head_dim = d_model // num_heads

        V = V.view(-1, num_heads, head_dim)  # [key_pos, head, head_dim]
        attn = attn.permute(1, 2, 0)  # [pos, key_pos, head]

        output = torch.einsum("p k h, k h d -> p k h d", attn, V)  # [pos, key_pos, head, head_dim]
        output = output.reshape(attn.shape[1], attn.shape[2], num_heads, head_dim)  # sanity step
        output = output.view(attn.shape[1], attn.shape[2], num_heads, head_dim)  # [pos, key_pos, head, head_dim]
        output = output.permute(2, 0, 1, 3).reshape(num_heads, attn.shape[1], attn.shape[2], head_dim)  # intermediate sanity

        # Expand dims and repeat to get full d_model size
        output = output.reshape(attn.shape[1], attn.shape[2], num_heads * head_dim)
        output = output.unsqueeze(0)  # [batch, pos, key_pos, d_model]

        # Add back the head dimension
        output = output.view(1, attn.shape[1], attn.shape[2], num_heads, head_dim * 1)
        return cast(Float[Tensor, "batch pos key_pos head d_model"], output.unsqueeze(0))