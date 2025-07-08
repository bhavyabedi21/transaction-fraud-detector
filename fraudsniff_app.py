
import numpy as np
import pandas as pd
import streamlit as st
import joblib

with open('final_model.joblib', 'rb') as file:
    model = joblib.load(file)


def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]
    chance = round(pred * 100, 2)
    if pred > 0.5:
        return f'This transaction is more likely to be fraudulent, chances {chance}%'
    else:
        return f'This transaction is less likely to be fraudulent, chances {chance}%'

def main():
    st.title('FRAUDSNIFF - Transaction Fraud Detector')

    day = st.slider('Enter the day of the month on which the transaction occurred', min_value=1, max_value=31, step=1)
    hour = st.slider('At what hour of the day did the transaction occur? (Use 24-hour format, e.g., 14 for 2 PM)', min_value=1, max_value=24, step=1)
    type_trans = st.selectbox('Select the type of transaction', ['Cash In', 'Cash Out', 'Debit', 'Payment', 'Transfer'])
    amt = st.text_input('Enter the total amount of transaction')
    oldbalorg = st.text_input('Enter the balance of the account of sender before transaction')
    newbalorg = st.text_input('Enter the balance of the account of sender after transaction')
    oldbaldes = st.text_input('Enter the balance of the account of receiver before transaction')
    newbaldes = st.text_input('Enter the balance of the account of receiver after transaction')

    type_trans_cashin = 0
    type_trans_cashout = 0
    type_trans_debit = 0
    type_trans_payment = 0
    type_trans_transfer = 0

    if type_transfer == 'Cash In':
        type_trans_cashin = 1
    elif type_transfer == 'Cash Out':
        type_trans_cashout = 1
    elif type_transfer == 'Debit':
        type_trans_debit = 1
    elif type_transfer == 'Payment':
        type_trans_payment = 1
    else:
        type_trans_transfer = 1
        

    if st.button('Predict'):
        if not amt or not oldbalorg or not newbalorg or not oldbaldes or not newbaldes:
            st.error("Please fill in all numeric fields")
            return

        try:
            amt_val = float(amt)
            oldbalorg_val = float(oldbalorg)
            newbalorg_val = float(newbalorg)
            oldbaldes_val = float(oldbaldes)
            newbaldes_val = float(newbaldes)
        except ValueError:
            st.error("Enter valid numbers in numeric fields")
            return

        inp_list = [day, hour, type_trans_cashin, type_trans_cashout, type_trans_debit, type_trans_payment,
                    type_trans_transfer, amt_val, oldbalorg_val, newbalorg_val, oldbaldes_val, newbaldes_val]

        response = prediction(inp_list)
        st.success(response)

if __name__ == '__main__':
    main()
