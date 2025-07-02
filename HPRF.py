import pandas as pd
#cargar los datos de entrenamiento y definirlos
df = pd.read_csv("train.csv")
X = df.drop(columns=["SalePrice"])
y = df.SalePrice
#importar modulos
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
#definir las columnas que voy a codificar
encode_cols = ["Neighborhood","BsmtQual","KitchenQual","ExterQual",
              "GarageFinish","GarageType","FireplaceQu","Foundation"]
#definir las columnas numericas  
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#definir el transformer para columnas numericas
num_transformer = Pipeline([
    ("Imputer", SimpleImputer(strategy='median')),
    ("Scaler", StandardScaler())
])
#definir el transformer para columnas categoricas
cat_transformer = Pipeline([
    ("Imputer", SimpleImputer(strategy='most_frequent')),
    ("Encoder", TargetEncoder(smoothing=1.0, handle_unknown='ignore'))
])
#juntar los 2 transformers
transformer = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, encode_cols)
], remainder='drop') #con "remainder" definimos qué hacer con las columnas no transformadas
#definimos el modelo final

model = Pipeline([
    ("Preprocessor", transformer),
    ("model", RandomForestRegressor(n_estimators=200,
                                    random_state=52))
])
#entrenamos el modelo
model.fit(X, y)
#cargamos los datos de prueba
test_data = pd.read_csv("test.csv")
#predecimos
wa = model.predict(test_data)
#convertimos a DataFrame
output = pd.DataFrame({'Id': test_data['Id'],
                       'SalePrice':wa})
#creamos un archivo .csv con el DataFrame
output.to_csv("submission3.csv", index=False)