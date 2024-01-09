import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from torch.nn import Module, Sequential, Linear, Dropout, Tanh

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

class Classifer(Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.Classifer = Sequential(
            Linear(768, 256),
            Tanh(),
            Dropout(0.3),
            Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        bert_output = bert_output[1]
        output = self.Classifer(bert_output)
        return output


bert = AutoModel.from_pretrained("vinai/phobert-base")
model = Classifer(bert)
model.load_state_dict(torch.load("Lab6/Bai10.pt"))
model.eval()

def predict(model, text):
    input_ids = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True, return_tensors='pt')['input_ids']
    output = model(input_ids, None, None)
    _, predicted = torch.max(output, 1)
    return predicted

def Bai10():
    st.title("Bài 10")
    input_text = st.text_input("Nhập câu cần dự đoán", "")
    if input_text != "":
        predicted = predict(model, input_text)
        if predicted == 0:
            st.write("Negative")
        elif predicted == 1:
            st.write("Neutral")
        else:
            st.write("Positive")
            