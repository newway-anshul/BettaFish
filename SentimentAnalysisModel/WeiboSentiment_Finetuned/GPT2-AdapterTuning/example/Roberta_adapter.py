from transformers.models.roberta.modeling_roberta import RobertaLayer

class RobertaLayerWithAdapter(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        # Assuming adapter size is 64
        adapter_size = 64
        self.adapter = AdapterLayer(config.hidden_size, adapter_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        # Call the original forward pass method
        self_outputs = super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        # Get the output from Transformer layer
        sequence_output = self_outputs[0]
        # Pass the output through adapter layer
        sequence_output = self.adapter(sequence_output)
        # Return the modified output (other outputs remain unchanged)
        return (sequence_output,) + self_outputs[1:]

"""
Each RobertaLayer in RoBERTa contains a self-attention mechanism and a feed-forward network, which jointly constitute RoBERTa's basic architecture.
"""
