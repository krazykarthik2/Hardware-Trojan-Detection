import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def create_numerics(data):
    # Get nominal columns
    nominal_cols = data.select_dtypes(include='object').columns.tolist()

    # Turn nominal to numeric
    for nom in nominal_cols:
        enc = LabelEncoder()
        enc.fit(data[nom])
        data[nom] = enc.transform(data[nom])
        print( nom, data[nom])


    return data

def prepare_predict_data(): # clean the code afterwards when i have time
    data = pd.read_excel("Data/input.xlsx")
    # take input data from excel 
    # take scaler model  adjust data to the model 
    with open('scalers/minmax_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("scaler loaded")
    # filter out that those are not in the scaler
    data = data[data.columns.intersection(scaler.feature_names_in_)]
    data = scaler.transform(data)

    # return input data
    return data
    
    
def prepare_data():
    data = pd.read_excel("Data/HEROdata2.xlsx")
    data = data.dropna()
    trojan_free = data.loc[data['Label']=="'Trojan Free'"].reset_index()    
    
    # balance the ratio between trojan free and infected of the same circuit category
    for i in range(len(trojan_free)):
        category_substring = trojan_free['Circuit'][i].replace("'",'')
        circuit_group = data['Circuit'].filter(like=category_substring, axis=0)
        # circuit_group = data[data['Circuit'].str.contains(category_substring, na=False, case=False)]
        
        df1 = circuit_group.iloc[0:1]
        
        if len(circuit_group) > 1:
            data = data.add_suffix([df1]*(len(circuit_group)-1))
            #changed from append to add_suffix and deleted ,ignore_index=True
            # ! IMPORTANT CHANGE !
    
    data.drop(columns=['Circuit'], inplace=True)

    data = create_numerics(data)
    
    data = shuffle(data, random_state=42)

    # Create correlation matrix
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                      k=1).astype(np.integer) == 1)

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features
    data = data.drop(data[to_drop], axis=1)

    
    y = pd.DataFrame(data["Label"]).values
    x = data.drop(["Label"], axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    
    with open('scalers/minmax_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to disk as minmax_scaler.pkl")
    
    """
    # plot the correlated features
    sns.heatmap(
        corr_matrix,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    plt.title("Features correlation")
    plt.show()
    """
    return(x_train, x_test, y_train, y_test)