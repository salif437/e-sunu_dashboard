import streamlit as st
import pandas as pd
import plotly.express as px
#from streamlit_option_menu import option_menu
#from numerize.numerize import numerize
import time
import os
#import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import plotly.graph_objs as go
from tabulate import tabulate


#from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(page_title="Dashboard",page_icon="🛒",layout="wide")

# le logo
st.sidebar.image("image/logo.png",caption="")

# le titre de la page
st.title("TABLEAU DE BORD DE e-SUNU SHOP")

# petite description

st.markdown("""
Ce tableau de bord vous permet d'explorer les données commerciales à travers divers graphiques interactifs.
Utilisez les filtres et les graphiques pour obtenir des insights précieux sur le comportement des clients, les performances des produits.
""")

theme_plotly = None 

# ouvrir le fichier css
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)



# ----------------------IMPORTATION DES DONNEES------------------------------
@st.cache #_data
def importation(path):
    return pd.read_csv(path, encoding='ISO-8859-1')

# customers
print("TABLE CUSTOMERS")
cst_path = "./datasets/Customers.csv"
customer = importation(cst_path)

# exchange
print("TABLE EXCHANGE")
exc_path= "./datasets/Exchange_Rates.csv"
exchange = importation(exc_path)

# product
print("TABLE PRODUCT")
prod_path= "./datasets/Products.csv"
product = importation(prod_path)

# sales
print("TABLE SALES")
sls_path= "./datasets/Sales.csv"
sales = importation(sls_path)

# store
print("TABLE STORE")
str_path= "./datasets/Stores.csv"
store = importation(str_path)


# ----------------------RENOMMER LES COLONNES------------------------------


customer= customer.rename(columns={"CustomerKey": "customer_id","Gender": "gender", "Name": "name", "City":"cst_city","State Code": "cst_state_code","State" : "cst_state","Zip Code":"cst_zip_code","Country" : "cst_country","Continent": "cst_continent","Birthday":"cst_birthday"})

# exchange
exchange = exchange.rename(columns={"Date": "exchange_date", "Currency":"currency_code", "Exchange":"exchange"})

# product
product = product.rename(columns={"ProductKey": "product_id","Product Name": "product_name",
"Brand": "brand", "Color": "color", "Unit Cost USD": "unit_cost_USD","Unit Price USD": "unit_price_USD", 
"SubcategoryKey": "subcategory_id","Subcategory": "subcategory", "CategoryKey": "category_id", "Category":"category"})

# sales
sales = sales.rename(columns={"Order Number" : "order_number","Line Item" : "line_item", "Order Date":"order_date",
"Delivery Date":"delivery_date", "CustomerKey": "customer_id" , "StoreKey":"store_id" ,"ProductKey": "product_id",
"Quantity":"quantity", "Currency Code":"currency_code"})

# store
store = store.rename(columns={"StoreKey":"store_id","State":"store_state", "Square Meters":"square_meters", "Open Date":"open_date", "Country":"st_country"})


# ----------------------MODIFICATION DES TYPES DE DONNEES NECESSAIRES------------------------------


# customer
customer['customer_id'] = customer['customer_id'].astype(str)

customer['cst_birthday'] = pd.to_datetime(customer['cst_birthday'], format='%m/%d/%Y')

# exchange
exchange['exchange_date']  = pd.to_datetime(exchange['exchange_date'] )
exchange['exchange_date']  = exchange['exchange_date'] .dt.strftime('%m/%d/%Y')

exchange['exchange'] = exchange['exchange'].astype(str).str.replace('.', ',')
exchange['exchange'] = exchange['exchange'].str.replace(',', '.').astype(float)


# product

product['product_id'] = product['product_id'].astype(str)

product['unit_cost_USD'] = product['unit_cost_USD'].astype(str).str.replace('$', '')
product['unit_cost_USD'] = product['unit_cost_USD'].astype(str).str.replace(',', '')
product['unit_cost_USD'] = product['unit_cost_USD'].astype(str).str.replace('.', ',')
product['unit_cost_USD'] = product['unit_cost_USD'].str.replace(',', '.').astype(float)

product['unit_price_USD'] = product['unit_price_USD'].astype(str).str.replace('$', '')
product['unit_price_USD'] = product['unit_price_USD'].astype(str).str.replace(',', '')
product['unit_price_USD'] = product['unit_price_USD'].astype(str).str.replace('.', ',')
product['unit_price_USD'] = product['unit_price_USD'].str.replace(',', '.').astype(float)

            # enlever le nom des marques sur le nom du produit
product['product_name'] = product['product_name'].astype(str).str.replace('Contoso', '')
product['product_name'] = product['product_name'].astype(str).str.replace('WWI', '')
product['product_name'] = product['product_name'].astype(str).str.replace('NT', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Adventure Works', '')
product['product_name'] = product['product_name'].astype(str).str.replace('SV', '')
product['product_name'] = product['product_name'].astype(str).str.replace('A. Datum', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Fabrikam', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Litware ', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Proseware', '')
product['product_name'] = product['product_name'].astype(str).str.replace('MGS ', '')
product['product_name'] = product['product_name'].astype(str).str.replace('The Phone Company', '')

            # enlever les couleurs dans le nom du produit
product['product_name'] = product['product_name'].astype(str).str.replace('Azure', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Black', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Blue', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Brown', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Grey', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Gold', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Green', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Orange', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Pink', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Purple', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Red', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Silver', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Silver Grey', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Transparent', '')
product['product_name'] = product['product_name'].astype(str).str.replace('White', '')
product['product_name'] = product['product_name'].astype(str).str.replace('Yellow', '')

product['subcategory_id'] = product['subcategory_id'].astype(str)

product['category_id'] = product['category_id'].astype(str)


# sales

sales['order_number'] = sales['order_number'].astype(str)

sales['order_date'] = pd.to_datetime(sales['order_date'])

sales['delivery_date'] = pd.to_datetime(sales['delivery_date'])

sales['customer_id'] = sales['customer_id'].astype(str)

sales['store_id'] = sales['store_id'].astype(str)

sales['product_id'] = sales['product_id'].astype(str)


# store
store['store_id'] = store['store_id'].astype(str)

store['square_meters'] = store['square_meters'].astype(str).str.replace('.', ',')
store['square_meters'] = store['square_meters'].str.replace(',', '.').astype(float)

store['open_date'] = pd.to_datetime(store['open_date'])


# ----------------------JOINTURE DES DATASETS------------------------------


df_1 = pd.merge(sales, customer, how= "inner", on='customer_id')

df_2 = pd.merge(df_1, store,how= "inner", on='store_id')

df = pd.merge(df_2, product,how= "inner", on='product_id')

df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

# Extraire l'année
df['order_year'] = df['order_date'].dt.year
df['order_date'] = pd.to_datetime(df['order_date'])



# Extraire le mois de la commande
order_month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['order_month'] = df['order_date'].dt.strftime('%B')
df['month'] = pd.Categorical(df['order_month'], categories=order_month, ordered=True)


# ----------------------CREATION DES GRAPHIQUES------------------------------

# Les filtres

selected_country=st.sidebar.multiselect(
    "Sélectionnez un ou plusieurs pays",
     options=df['cst_country'].unique(),
)

selected_category=st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs catégories",
     options=df['category'].unique(),
)

selected_color =st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs couleurs",
     options=df["color"].unique(),
)

selected_year =st.sidebar.multiselect(
    "Sélectionnez un ou plusieurs années",
     options=df["order_year"].unique(),
)

selected_month=st.sidebar.multiselect(
    "Sélectionnez un ou plusieurs mois",
     options=df["month"].unique(),
)

df_selection = df[
    (df['category'].isin(selected_category) if selected_category else df['category'].notnull()) &
    (df['month'].astype(str).isin(selected_month) if selected_month else df['month'].notnull()) &
    (df['color'].isin(selected_color) if selected_color else df['color'].notnull())&
    (df['order_year'].isin(selected_year) if selected_year else df['order_year'].notnull())&
    (df['cst_country'].isin(selected_country) if selected_country else df['cst_country'].notnull())
]


# Calculs préliminaires

df['delivery_delay'] = df['delivery_date'].fillna(pd.to_datetime('NaT')) - df['order_date']
df['delivery_delay'] = df['delivery_delay'].dt.days.fillna(-1)  # Remplacer NaT par -1

mean_delivery_delay = df[df['delivery_delay'] > -1]['delivery_delay'].mean() 
df['delivery_date'] = df.apply(lambda row: row['order_date'] + pd.Timedelta(days=mean_delivery_delay) if row['delivery_delay'] == -1 else row['delivery_date'],
    axis=1)

print(f"Le temps de livraison moyen est de {mean_delivery_delay:.2f} jours.")


# Vérifier si la colonne 'total_sales' est présente dans 'df_selection'

if 'total_sales' not in df_selection.columns:
    df_selection['total_sales'] = df_selection['quantity'] * df_selection['unit_price_USD']

# Calcul du chiffre d'affaires total

total_sales = df_selection['total_sales'].sum()
qty_sold = float(df_selection['quantity'].sum())
top_category = df_selection['category'].mode().to_string(index=False)
top_store = df_selection['st_country'].mode().to_string(index=False)


#définir une fonction "Home"
def Home():

    if 'delivery_delay' not in df_selection.columns:
        df_selection['delivery_delay'] = df_selection['delivery_date'].fillna(pd.to_datetime('NaT')) - df_selection['order_date']
        df_selection['delivery_delay'] = df_selection['delivery_delay'].dt.days
        mean_delivery_delay = df_selection[df_selection['delivery_delay'] >= 0]['delivery_delay'].mean()
        df_selection['delivery_delay'] = df_selection['delivery_delay'].apply(lambda x: mean_delivery_delay if x < 0 else x)
    average_delivery_time = df_selection['delivery_delay'].mean()

    # Affichage des KPI
    total1, total2, total3, total4 = st.columns(4, gap="small")

    with total1:
        st.info('Ventes totales', icon="📈")
        st.metric(label='', value=f"$ {total_sales:,.0f}")

    with total2:
        st.info('Quantité vendue', icon="🛒")
        st.metric(label='', value=f"{qty_sold:,.0f}")

    with total3:
        st.info('Catégorie supérieure', icon="🏢")
        st.metric(label='', value=top_category)

        
    with total4:
        st.info('Temps moyen de livraison ', icon="🚚")
        st.metric(label='', value=f"{average_delivery_time:.2f} jours")
    #style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow="#F71938")


def graphs():
    # Création de trois colonnes
    col1, col2, col3 = st.columns(3)
   
    title_style = """
    <style>
    .title {
        font-family: 'Arial', sans-serif; /* Changer la police ici */
        font-size: 18px; /* Changer la taille de la police ici */
        color: #333; /* Changer la couleur de la police ici */
        margin-bottom: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;

   }
    </style>"""
    st.markdown(title_style, unsafe_allow_html=True)

    with col1:
        st.markdown("<h4 class='centered'>Commandes par jour de la semaine</h4>", unsafe_allow_html=True)

        df['day_of_week'] = df_selection['order_date'].dt.day_name()
        day_of_week_orders = df['day_of_week'].value_counts().reset_index()
        day_of_week_orders.columns = ['day_of_week', 'orders']

        fig = px.bar(
            day_of_week_orders,
            x='day_of_week',
            y='orders',
            color='orders',  # Utilisez 'orders' pour la coloration des barres
            color_continuous_scale=px.colors.sequential.Viridis,  # Palette de couleurs
            labels={'day_of_week': 'Jour de la semaine', 'orders': 'Commandes'},
        )
        # Mise à jour de la mise en page pour masquer la légende
        fig.update_layout(
            showlegend=False  # Masquer la légende
        )
        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.markdown("<h4 class='centered'>Proportion des ventes par Pays</h4>", unsafe_allow_html=True)
        
        sales_by_country = df_selection.groupby(['st_country'])['total_sales'].sum()
        sales_by_country = sales_by_country.sort_values()
        
        fig_growth = px.pie(
            sales_by_country,
            values=sales_by_country.values,
            names=sales_by_country.index,
            template='plotly_white',
            color_discrete_sequence=px.colors.sequential.Viridis,

        )
        
        fig_growth.update_layout(
            legend_title="Pays",
            margin=dict(t=5, b=5, l=5, r=5),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.28,
                xanchor="right",
                x=0.95
            ),
            
        )
        with st.container():
            st.plotly_chart(fig_growth, use_container_width=True)
    
    with col3:
        st.markdown("<h4 class='centered'>Top 10-produits les plus vendus</h4>", unsafe_allow_html=True)
        top_products = df_selection.groupby('product_name')['total_sales'].sum().nlargest(10)
        top_products.columns = ['Produit', 'Somme des ventes'] 
        st.dataframe(top_products)



def graphs2():

    col1, col2 = st.columns(2)
   
    title_style = """
    <style>
    .title {
        font-family: 'Arial', sans-serif; /* Changer la police ici */
        font-size: 18px; /* Changer la taille de la police ici */
        color: #333; /* Changer la couleur de la police ici */
        margin-bottom: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;

   }
    </style>"""


    with col1:
        
        with st.container():
            st.markdown("<h4 class='centered'>Répartition du chiffre d'affaires par tranche d'âge</h4>", unsafe_allow_html=True)
            today = pd.to_datetime('today')
            df_selection['age'] = today.year - df_selection['cst_birthday'].dt.year
            bins = [0, 18, 25, 35, 45, 55, 65, 100]
            labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            df_selection['age_group'] = pd.cut(df_selection['age'], bins=bins, labels=labels, right=False)

            age_distribution = df_selection.groupby('age_group', observed=False)['total_sales'].sum().reset_index()

            # Créer un graphique à barres avec Plotly
            fig = px.bar(
                age_distribution,
                x='age_group',
                y='total_sales',
                color='age_group',  # Utilise la colonne 'age_group' pour la coloration
                color_discrete_sequence=px.colors.sequential.Viridis,  # Choisissez une palette de couleurs
                labels={'age_group': 'Tranche d\'âge', 'total_sales': 'Chiffre d\'affaires (USD)'},
            )
            
            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Titre de l'application
        st.markdown("<h4 class='centered'>Part de chaque marque dans le chiffre d\'affaires</h4>", unsafe_allow_html=True)

        # Extraire l'année
        df['delivery_year'] = df['delivery_date'].dt.year
        df['delivery_year'] = df['delivery_date'].dt.year
        # Filtrer df en fonction des années sélectionnées

        if 'delivery_year' not in df_selection.columns:
            df_selection['delivery_year'] = df['delivery_date'].dt.year

        
        # Agréger les ventes totales par marque et par année
        annual_brand_sales = df_selection.groupby(['delivery_year', 'brand'])['total_sales'].sum().reset_index()

        # Créer le graphique à barres empilées horizontal avec Plotly
        fig = px.bar(
            annual_brand_sales,
            x='total_sales',
            y='delivery_year',
            color='brand',
            orientation='h',
            title='',
            labels={'total_sales': 'Chiffre d\'affaires (USD)', 'delivery_year': 'Année'},
            color_discrete_sequence=px.colors.sequential.Viridis,
            height=600
        )

        # Personnaliser la légende en haut
        fig.update_layout(
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1,
                xanchor='right',
                x=1
            ),
            xaxis_title='Chiffre d\'affaires (USD)',
            yaxis_title='Année',
            barmode='stack',
            title={
                'text': '',
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color='black', family='Arial')
            },
        )
            # Affichage du graphique dans Streamlit
        with st.container():
            st.plotly_chart(fig, use_container_width=True)
        



def graphs3():
    title_style = """
    <style>
    .title {
        font-family: 'Arial', sans-serif; /* Changer la police ici */
        font-size: 18px; /* Changer la taille de la police ici */
        color: #333; /* Changer la couleur de la police ici */
        margin-bottom: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;

   }

    </style>"""
    st.markdown("<h4 class='centered'>Évolution des ventes par année et prévision</h4>", unsafe_allow_html=True)
    
    # Vérifiez si une seule année est sélectionnée
    selected_years = df_selection['order_year'].unique()
    if len(selected_years) == 1:
        st.error("Veuillez sélectionner plusieurs années pour voir l'évolution des ventes et les prévisions.")
        return

    # Calculer les ventes totales par année
    annual_sales = df_selection.groupby('order_year')['total_sales'].sum().reset_index()

    # Préparer les données pour la régression linéaire
    X = np.array(annual_sales['order_year']).reshape(-1, 1)
    y = annual_sales['total_sales'].values
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    # Prévisions pour les 5 prochaines années
    forecast_periods = 5
    forecast_years = np.array(range(annual_sales['order_year'].max() + 1, annual_sales['order_year'].max() + 1 + forecast_periods)).reshape(-1, 1)
    forecast_years_b = np.c_[np.ones((forecast_years.shape[0], 1)), forecast_years]
    forecast = forecast_years_b.dot(theta_best)

    # Créer un DataFrame pour les prévisions
    forecast_df = pd.DataFrame({
        'order_year': forecast_years.flatten(),
        'total_sales': forecast
    })
    
    # Tracer les prévisions
    combined_df = pd.concat([ annual_sales,forecast_df] )

    # Graphique avec Plotly
    fig = px.line(
        combined_df,
        x='order_year',
        y='total_sales',
        markers=True,
        labels={'order_year': 'Année', 'total_sales': 'Chiffre d\'affaires (USD)'}
    )
    # Afficher les prévisions sous forme de tableau
    st.write(forecast_df)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

# menu pour affichage des graphiques
def sideBar():
    Home()
    graphs()
    graphs2()
    graphs3()

sideBar()



hide_st_style=""" 

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""