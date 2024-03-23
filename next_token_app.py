import streamlit as st
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generating the next characters with the current model by passing the context size, like sliding window of sorts
def generate_next_chars(model, stoi, itos, context, k):
    context = [stoi[ch] for ch in context]

    generated_tokens = []
    g = torch.Generator()
    g.manual_seed(seed)
    for _ in range(k):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        generated_tokens.append(ch)
        context = context[1:] + [ix]  
    return ''.join(generated_tokens)

st.title("Next Token Prediction")

# Inputting values from the users in the app
emb_size = st.slider("Embedding Size", 2, 6, value=4)
block_size = st.slider("Block Size", 5, 10, value=7)
model_num = st.slider("Model Number", 1, 5, value =3)
context = st.text_input("Context String", value="brutus:", max_chars=block_size)
seed = st.number_input("Random Seed", min_value=1, value=100)
k = st.number_input("Number of Characters to Generate", min_value=1, value=100)

# Defining all the models, so as to load the pre-trained models' .pth files from /models
if(model_num == 1):
    class NextChar(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size // 2)
            self.lin3 = nn.Linear(hidden_size // 2, vocab_size)

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = torch.sin(self.lin1(x))
            x = torch.relu(self.lin2(x))
            x = self.lin3(x)
            return x
        
elif(model_num == 2 or model_num == 3):
    class NextChar(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size // 2)
            self.lin3 = nn.Linear(hidden_size//2, hidden_size // 2)
            self.lin4 = nn.Linear(hidden_size // 2, vocab_size)


        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = torch.sin(self.lin1(x))
            x = torch.relu(self.lin2(x))
            x = torch.sin(self.lin3(x))
            x = self.lin4(x)

            return x
        
elif(model_num == 5):
    class NextChar(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, int(hidden_size*1.5))
            self.lin2 = nn.Linear(int(hidden_size*1.5), hidden_size)
            self.lin3 = nn.Linear(hidden_size, hidden_size // 2)
            self.lin4 = nn.Linear(hidden_size // 2, vocab_size)


        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = torch.sin(self.lin1(x))
            x = torch.relu(self.lin2(x))
            x = torch.sin(self.lin3(x))
            x = self.lin4(x)

            return x
        
elif(model_num == 4):
    class NextChar(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, int(hidden_size*1.5))
            self.lin2 = nn.Linear(int(hidden_size*1.5), hidden_size )
            self.lin3 = nn.Linear(hidden_size, hidden_size // 2)
            self.lin4 = nn.Linear(hidden_size // 2, vocab_size)


        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = torch.sin(self.lin1(x))
            x = torch.relu(self.lin2(x))
            x = torch.sin(self.lin3(x))
            x = self.lin4(x)

            return x

# Defining the size and then creating a NextChar instance as defined by the model_num
def load_model(emb_size, block_size, model_num, vocab_size):
    model_path = f"models/model_{emb_size}_{block_size}_{model_num}.pth"
    if model_num == 1: 
        hidden_size = 100
    elif model_num == 2: 
        hidden_size = 300
    elif model_num == 3: 
        hidden_size = 100
    elif model_num == 4: 
        hidden_size = 400
    elif model_num == 5: 
        hidden_size = 700
    
    model = NextChar(block_size, vocab_size, emb_size, hidden_size)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith('_orig_mod.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    model.eval()
    return model

# Generate but to generate next characters as the button is pressed
if st.button("Generate"):
    itos = {0: ' ', 1: '!', 2: '&', 3: "'", 4: ',', 5: '-', 6: '.', 7: ':', 8: ';', 9: '?', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'}
    stoi = {i:s for s,i in itos.items()}
    vocab_size = 36
    model = load_model(emb_size, block_size, model_num, vocab_size)

    next_chars = generate_next_chars(model, stoi, itos, context, k)
    st.write("Next characters:")
    st.write(next_chars)
