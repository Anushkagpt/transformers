import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class LCSModel(nn.Module):
    def __init__(self, model_name_or_path, tokenizer_name_or_path):
        super(LCSModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def forward(self, input_seq1, input_seq2):
        # Tokenize the input sequences
        inputs1 = self.tokenizer.encode_plus(
            input_seq1,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )
        inputs2 = self.tokenizer.encode_plus(
            input_seq2,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Create a batch with the two input sequences
        batch = {'input_ids': torch.cat((inputs1['input_ids'], inputs2['input_ids']), dim=0),
                 'attention_mask': torch.cat((inputs1['attention_mask'], inputs2['attention_mask']), dim=0)}

        # Forward pass through the transformer model
        outputs = self.model(**batch)

        # Extract the last hidden state of the transformer model
        last_hidden_state = outputs.last_hidden_state[:, 0, :]

        # Compute the similarity between the two input sequences
        similarity = F.cosine_similarity(last_hidden_state[0], last_hidden_state[1])

        # Predict the LCS using the similarity score
        lcs = self.predict_lcs(input_seq1, input_seq2, similarity)

        return lcs

    def predict_lcs(self, input_seq1, input_seq2, similarity):
        # Implement a simple LCS prediction algorithm using the similarity score
        # For example, you can use a threshold-based approach
        if similarity > 0.5:
            # Predict the LCS using a simple algorithm (e.g., dynamic programming)
            lcs = self.dynamic_programming_lcs(input_seq1, input_seq2)
        else:
            lcs = ""

        return lcs

    def dynamic_programming_lcs(self, input_seq1, input_seq2):
        # Implement a dynamic programming algorithm to find the LCS
        # This is a simple example, you can improve it using more advanced techniques
        m = len(input_seq1)
        n = len(input_seq2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if input_seq1[i - 1] == input_seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if input_seq1[i - 1] == input_seq2[j - 1]:
                lcs.append(input_seq1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return "".join(reversed(lcs))
