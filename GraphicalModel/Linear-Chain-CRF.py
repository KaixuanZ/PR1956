import torch
from torch import nn

class LCCRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).
    Args:
        nb_labels (int): number of labels in your tagset, including special symbols.
        bos_tag_id (int): integer representing the beginning of sentence symbol in
            your tagset.
        eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
        batch_first (bool): Whether the first dimension represents the batch dimension.
    """

    def __init__(
        self, nb_labels, bos_tag_id, eos_tag_id, batch_first=True
    ):
        super().__init__()
        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # enforce contraints (rows=from, columns=to) with a big negative number
        # so exp(-10000) will tend to zero
        # e.g. no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        # e.g. no transition alloed from the end of sentence
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

    def forward(self, emissions, tags, mask=None):
        """Compute the negative log-likelihood. See `log_likelihood` method."""
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
        Returns:
            torch.Tensor: the log-likelihoods for each sequence in the batch.
                Shape of (batch_size,)
        """

        # fix tensors order by setting batch as the first dimension
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)