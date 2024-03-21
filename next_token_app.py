import streamlit as st
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate the next k characters
def generate_next_chars(model, stoi, itos, context, k):
    # Convert context string to list of indices
    context = [stoi[ch] for ch in context]

    generated_tokens = []
    for _ in range(k):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        generated_tokens.append(ch)
        context = context[1:] + [ix]  # Update context with the generated token
    return ''.join(generated_tokens)

# Streamlit app
st.title("Next Token Prediction")

emb_size = st.slider("Embedding Size", 2, 6, value=4)
block_size = st.slider("Block Size", 5, 10, value=7)
model_num = st.slider("Model Number", 1, 5, value =3)
context = st.text_input("Context String", value="brutus: ", max_chars=block_size)
k = st.number_input("Number of Characters to Generate", min_value=1, value=100)

# Define the model architecture (this should match your saved models)
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

# Function to load a model
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
    
    # Choose the correct class based on model_num
    model = NextChar(block_size, vocab_size, emb_size, hidden_size)
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Handling for DataParallel state dicts or custom naming conventions
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith('_orig_mod.') else k  # Remove prefix
        new_state_dict[name] = v
    
    # Load the modified state dict
    model.load_state_dict(new_state_dict)
    
    model.eval()
    return model


if st.button("Generate"):
    # Load the model
    #0  # Adjust this based on your model# Adjust this based on your model
    itos = {0: ' ',
 1: '!',
 2: '&',
 3: "'",
 4: ',',
 5: '-',
 6: '.',
 7: ':',
 8: ';',
 9: '?',
 10: 'a',
 11: 'b',
 12: 'c',
 13: 'd',
 14: 'e',
 15: 'f',
 16: 'g',
 17: 'h',
 18: 'i',
 19: 'j',
 20: 'k',
 21: 'l',
 22: 'm',
 23: 'n',
 24: 'o',
 25: 'p',
 26: 'q',
 27: 'r',
 28: 's',
 29: 't',
 30: 'u',
 31: 'v',
 32: 'w',
 33: 'x',
 34: 'y',
 35: 'z'}  # Your character to index mapping
    stoi = {i:s for s,i in itos.items()}  # Your index to character mappingoi)
    print(stoi)
    vocab_size = 36
    model = load_model(emb_size, block_size, model_num, vocab_size)

    # Define your stoi and itos mappings

    # Generate the next k characters
    next_chars = generate_next_chars(model, stoi, itos, context, k)
    st.write("Next characters:", next_chars)
