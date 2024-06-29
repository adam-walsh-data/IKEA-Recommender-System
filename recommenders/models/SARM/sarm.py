import torch
import random
import torch.nn as nn

class MultiObjectiveQNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim,
        item_num,
        state_size,
        action_dim,
        gru_layers,
        embedding_dim=50,
        random_embed_init=True,
        train_pad_embed=True,
        use_packed_seq=False,
        padding_idx=None,
        name="QNetwork",
    ):
        super(MultiObjectiveQNetwork, self).__init__()
        self.state_size = state_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.item_num = item_num
        self.random_embed_init = random_embed_init
        self.use_packed_seq = use_packed_seq
        self.action_dim = action_dim
        self.gru_layers = gru_layers
        self.name = name

        # Item-Embeddings
        if random_embed_init:
            if padding_idx is None:
                padding_idx = self.item_num

            if use_packed_seq:
                train_pad_embed = True

            self.embedding = nn.Embedding(
                num_embeddings=self.item_num + 1,
                embedding_dim=self.embedding_dim,
                padding_idx=None if train_pad_embed else padding_idx,
            )
            self.embedding.weight.data.normal_(mean=0, std=0.01)

            if not train_pad_embed:
                with torch.no_grad():
                    self.embedding.weight[padding_idx] = torch.zeros(self.embedding_dim)
        else:
            raise NotImplementedError("TODO: Pretrained embeddings.")

        self.base_model = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        self.q_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.action_dim) for _ in range(5)
        ])

    def forward(self, s, lengths):
        if self.use_packed_seq:
            emb_seq = self.embedding(s)
            emb_seq = nn.utils.rnn.pack_padded_sequence(
                emb_seq, lengths=lengths, batch_first=True, enforce_sorted=False
            )
        else:
            emb_seq = self.embedding(s)

        _, h = self.base_model(emb_seq)
        h = h[0, :, :]

        outputs = [head(h) for head in self.q_heads]
        return outputs

class SARM_trainer:
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        train_pad_embed,
        use_packed_seq,
        learning_rate,
        item_num,
        state_size,
        action_dim,
        gru_layers,
        device,
        padding_idx=None,
        torch_rand_seed=118,
        python_rand_seed=999,
    ):
        torch.manual_seed(torch_rand_seed)
        random.seed(python_rand_seed)

        self.network = MultiObjectiveQNetwork(
            hidden_dim=hidden_dim,
            item_num=item_num,
            state_size=state_size,
            action_dim=action_dim,
            gru_layers=gru_layers,
            embedding_dim=embedding_dim,
            train_pad_embed=train_pad_embed,
            use_packed_seq=use_packed_seq,
            padding_idx=padding_idx,
        )

        self.device = device
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=None, reduction="mean")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.network.to(self.device)

    def train_step(self, s, a, r, s_next, true_len, true_next_len, is_end):
        r = r.unsqueeze(1)
        s, a, r, s_next = s.to(self.device), a.to(self.device), r.to(self.device), s_next.to(self.device)

        outputs = self.network(s, true_len)
        main_idx = random.randint(0, 4)
        DQN_main = outputs[main_idx]

        a_idx = a.unsqueeze(1)
        q = DQN_main.gather(1, a_idx)

        with torch.no_grad():
            outputs_next = self.network(s_next, true_next_len)
            max_a_next = torch.argmax(outputs_next[main_idx], dim=1).unsqueeze(1)
            q_next_boot = outputs_next[main_idx].gather(1, max_a_next)
            q_next_boot[is_end] = 0.0

        q_losses = []
        for i in range(5):
            q_loss = (r + self.gamma * outputs_next[i].gather(1, torch.argmax(outputs_next[i], dim=1).unsqueeze(1)) - outputs[i].gather(1, a_idx)) ** 2
            q_losses.append(torch.mean(q_loss))

        sup_loss = self.cross_entropy_loss(outputs[0], a)  # Supervised head loss

        # Calculate total loss as a scalarized combination
        q_head_weight = 1.0 / len(q_losses)
        total_q_loss = sum(q_losses) * q_head_weight
        loss_total = sup_loss + total_q_loss

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return sup_loss.item(), sum(q_loss.item() for q_loss in q_losses) / len(q_losses)

    def set_train(self):
        self.network.train()

    def set_eval(self):
        self.network.eval()

    def send_to_device(self):
        self.network.to(self.device)