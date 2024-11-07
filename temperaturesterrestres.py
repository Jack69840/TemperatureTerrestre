# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor


# Configuration de la page
st.set_page_config(
    page_title="Analyse de Température",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Définition des chemins pour les fichiers de données
data_path = "Data/"
files = {
    "df1": "Zonal annual means.csv",
    "df2": "Southern Hemisphere-mean monthly, seasonal, and annual means.csv",
    "df3": "Northern Hemisphere-mean monthly, seasonal, and annual means.csv",
    "df4": "Global-mean monthly, seasonal, and annual means.csv",
    "df5": "owid-co2-data.csv",
    "df_pib": "global-gdp-over-the-long-run.csv",
    "df_pop": "demo-pop-monde.xlsx"
}

# Chargement des fichiers dans des DataFrames
dataframes = {}
for key, file in files.items():
    file_path = os.path.join(data_path, file)
    if key == "df2" or key == "df3" or key == "df4":
        dataframes[key] = pd.read_csv(file_path, header=1)
    elif key == "df5":
        dataframes[key] = pd.read_csv(file_path, sep=",")
    elif file.endswith('.csv'):
        dataframes[key] = pd.read_csv(file_path)
    elif file.endswith('.xlsx'):
        dataframes[key] = pd.read_excel(file_path)

# Code pour le volet de navigation latéral
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une section", [
                            "Accueil", "Exploration des Données", "Visualisations", "Modèles et Prédictions"])

# Chemin relatif vers l'image dans le dossier "images"
image_path = os.path.join("images", "imagecouv.png")

# Vérification si l'image existe, puis affichage dans la barre latérale
if os.path.exists(image_path):
    st.sidebar.image(image_path, use_column_width=True)
else:
    st.sidebar.error(
        "L'image de couverture est introuvable. Assurez-vous qu'elle est dans le dossier 'images'.")

# Affichage du titre et des noms dans la barre latérale
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: #FFFFFF;">Analyse des Températures Terrestres</h2>
        <p style="color: #FFFFFF; font-size: 16px;">
            - Jacques Gauthier<br>
            - Thibault Le Boudec
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
#############################################################################
# Page Accueil
if page == "Accueil":
    st.markdown(
        """
        <h2 style="color: #FFFFFF;">La Hausse des Températures Terrestres : Une Crise Mondiale</h2>
        <p style="color: #FFFFFF; font-size: 18px;">
        Le climat terrestre, marqué par des cycles naturels de variation, connaît depuis la fin du XIXe siècle une hausse rapide des températures, sans précédent dans l’histoire humaine récente. Ce phénomène, souvent appelé "réchauffement climatique", est fortement lié à l'industrialisation qui a démarré il y a plus de 150 ans. À mesure que les activités humaines se sont intensifiées — extraction et combustion de combustibles fossiles, déforestation, industrialisation accrue — la quantité de dioxyde de carbone (CO₂), de méthane (CH₄) et d'autres gaz à effet de serre (GES) dans l'atmosphère a considérablement augmenté. Ces gaz piègent la chaleur, perturbant l’équilibre naturel du climat et entraînant une hausse des températures moyennes à l'échelle mondiale.
        </p>

        <h2 style="color: #FFFFFF;">L'Histoire et les Grandes Étapes de la Prise de Conscience Climatique</h2>
        <p style="color: #FFFFFF; font-size: 18px;">
        La prise de conscience mondiale de cette crise n'est pas immédiate. Dès les années 1970, les scientifiques alertent sur les effets potentiels des gaz à effet de serre, mais il faudra attendre les années 1990 pour que la question environnementale et climatique gagne en visibilité politique et médiatique. La première "Conférence des Parties" (COP), un rassemblement annuel de pays signataires de la Convention-cadre des Nations Unies sur les changements climatiques (CCNUCC), a eu lieu en 1995. Ce cadre marque le début d'un dialogue mondial structuré visant à définir des engagements collectifs pour réduire les émissions de gaz à effet de serre.
        </p>
        <p style="color: #FFFFFF; font-size: 18px;">
        Au fil des COP, plusieurs accords majeurs ont été conclus :
        </p>
        <ul style="color: #FFFFFF; font-size: 18px;">
            <li><strong>Protocole de Kyoto (1997)</strong> : Premier accord juridiquement contraignant, il impose des objectifs de réduction des émissions aux pays industrialisés.</li>
            <li><strong>Accord de Copenhague (2009)</strong> : Bien qu'il n'ait pas abouti à des engagements contraignants, cet accord reconnaît officiellement le besoin de limiter le réchauffement à 2°C.</li>
            <li><strong>Accord de Paris (2015)</strong> : Cet accord historique vise à contenir l’élévation des températures bien en dessous de 2°C, avec un objectif idéal de 1,5°C par rapport aux niveaux préindustriels. Il engage tous les pays à des contributions volontaires et revues périodiquement.</li>
        </ul>
        <p style="color: #FFFFFF; font-size: 18px;">
        Ces accords, bien qu’importants, posent des défis en termes de mise en œuvre et de suivi. Chaque COP est une occasion de réévaluer les engagements, d'ajuster les stratégies et d'inciter à des efforts accrus face aux impacts du réchauffement climatique.
        </p>

        <h2 style="color: #FFFFFF;">Notre Projet : Visualiser et Comprendre les Données Climatiques</h2>
        <p style="color: #FFFFFF; font-size: 18px;">
        À l'heure où la sensibilisation et la compréhension des enjeux climatiques deviennent essentielles, nous avons développé ce projet dans le but de rendre accessibles les données sur l'évolution des températures et les facteurs associés. Ce projet permet de suivre les tendances historiques des températures mondiales et de les relier à des facteurs explicatifs tels que les émissions de CO₂, la croissance démographique, et le PIB mondial. L’objectif est de fournir un outil qui aide à mieux comprendre l’impact de l’activité humaine sur le climat et, espérons-le, à renforcer l'engagement pour un avenir durable.
        </p>
        <p style="color: #FFFFFF; font-size: 18px;">
        Ce projet présente plusieurs visualisations interactives qui permettent d'explorer ces données et d'observer les corrélations entre les variables. Notre espoir est qu'en rendant ces données compréhensibles et accessibles, nous pourrons encourager une prise de conscience plus large et informer les discussions autour des actions à entreprendre pour atténuer les effets du réchauffement climatique.
        </p>

        <h2 style="color: #FFFFFF;">Conscience des Limites et Perspectives d'Amélioration</h2>
        <p style="color: #FFFFFF; font-size: 18px;">
        Bien que nous ayons utilisé les meilleures données disponibles, nous reconnaissons les limites de notre projet. Nos analyses se basent sur des données globales et historiques, et bien que les tendances soient claires, elles ne permettent pas de faire des prédictions précises sur des échelles plus locales ou à court terme. De plus, le réchauffement climatique est un phénomène complexe influencé par de nombreux facteurs qui ne sont pas tous couverts dans cette étude. Ce projet constitue néanmoins une première étape vers une meilleure compréhension des tendances globales et pourrait être étendu avec de nouvelles données et méthodes d’analyse plus avancées.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Bouton vers le site du GIEC
    if st.button("Visiter le site du GIEC"):
        st.write("[Site officiel du GIEC](https://www.ipcc.ch)")


############################################################################
# Page Exploration des Données
elif page == "Exploration des Données":
    st.markdown("### Exploration des Données")

    source = st.radio(
        "Sélectionnez une source de données",
        ["Source NASA", "Source OWID", "Source Banque Mondiale"],
        index=0
    )

    if source == "Source NASA":
        selected_dfs = ["df1", "df2", "df3", "df4"]
    elif source == "Source OWID":
        selected_dfs = ["df5"]
    elif source == "Source Banque Mondiale":
        selected_dfs = ["df_pib", "df_pop"]

    action = st.radio(
        "Affichage des données",
        ["Aperçu des 10 premières lignes", "Types de données", "Description des données",
            "Nombre de lignes et de colonnes", "Données manquantes ou erronées"],
        index=0
    )

    for key in selected_dfs:
        if action == "Aperçu des 10 premières lignes":
            st.write(dataframes[key].head(10))
        elif action == "Types de données":
            st.write(dataframes[key].dtypes)
        elif action == "Description des données":
            st.write(dataframes[key].describe())
        elif action == "Nombre de lignes et de colonnes":
            st.write(dataframes[key].shape)
        elif action == "Données manquantes ou erronées":
            st.write(dataframes[key].isnull().sum())

##########################################################################
# Chemin relatif vers le fichier de données
final_data_path = os.path.join("Data", "final_data.csv")

# Page Visualisations
if page == "Accueil":
    pass  # Placeholder to ensure structural integrity
elif page == "Visualisations":
    # Charger le fichier final_data.csv
    try:
        final_data = pd.read_csv(final_data_path)

        # Titre de la section
        st.markdown("## Visualisations")
        st.markdown(
            """
            Cette section présente des graphiques illustrant les tendances historiques des températures, 
            les relations entre la population et les émissions de CO₂, et le lien entre la croissance économique 
            et les émissions de gaz à effet de serre. Ces visualisations permettent de mieux comprendre les 
            dynamiques mondiales liées au réchauffement climatique.
            """
        )

        # Évolution des températures globales, hémisphère nord et sud (1880-2022)
        st.markdown(
            "### Évolution des Températures Globales, Hémisphère Nord et Sud (1880-2022)")

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=final_data['Year'], y=final_data['Glob'], mode='lines', name='Global', line=dict(color="red")))
        fig_temp.add_trace(go.Scatter(
            x=final_data['Year'], y=final_data['NHem'], mode='lines', name='Northern Hemisphere', line=dict(color="blue")))
        fig_temp.add_trace(go.Scatter(x=final_data['Year'], y=final_data['SHem'],
                           mode='lines', name='Southern Hemisphere', line=dict(color="green")))
        fig_temp.update_layout(
            title='Évolution des températures globales, hémisphère nord et sud (1880-2022)',
            xaxis_title='Année',
            yaxis_title='Température (°C)',
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        # Bouton pour afficher les commentaires
        if st.checkbox("Je veux en savoir plus sur ce graphe", key="temp"):
            st.markdown(
                """
                **Analyse :**
                Ce graphique montre l’évolution des températures moyennes à l'échelle globale et dans les hémisphères nord et sud depuis 1880. 
                Les tendances sont particulièrement frappantes après les années 1970, avec une montée rapide des températures globales. 
                On observe une élévation plus prononcée dans l’hémisphère nord (en bleu) par rapport à l'hémisphère sud (en vert), 
                probablement en raison d'une plus grande densité de terres habitées et industrialisées dans le nord, ce qui augmente 
                l’effet des activités humaines sur le climat. La tendance générale à la hausse confirme les impacts du réchauffement climatique.
                """
            )

        # Relation entre la population mondiale et les émissions de CO₂ (1880-2022)
        st.markdown(
            "### Relation entre la Population Mondiale et les Émissions de CO₂ (1880-2022)")

        fig_pop_co2 = go.Figure()

        # Ajout des deux courbes Population et CO₂ avec l'axe des années en abscisse
        fig_pop_co2.add_trace(go.Scatter(
            x=final_data['Year'],
            y=final_data['Population (millions)'],
            mode='lines',
            name="Population (millions)",
            line=dict(color="purple")
        ))
        fig_pop_co2.add_trace(go.Scatter(
            x=final_data['Year'],
            y=final_data['CO2 (kt)'],
            mode='lines',
            name="Émissions de CO₂ (kt)",
            line=dict(color="orange"),
            yaxis="y2"  # Utilisation d'un deuxième axe pour CO₂
        ))

    # Mise en forme des axes
        fig_pop_co2.update_layout(
            title='Relation entre la Population Mondiale et les Émissions de CO₂ (1880-2022)',
            xaxis=dict(title='Année'),
            yaxis=dict(title='Population (millions)',
                       titlefont=dict(color='purple')),
            yaxis2=dict(title='Émissions de CO₂ (kt)', titlefont=dict(
                color='orange'), overlaying='y', side='right')
        )
        st.plotly_chart(fig_pop_co2, use_container_width=True)

        # Bouton pour afficher les commentaires
        if st.checkbox("Je veux en savoir plus sur ce graphe", key="pop_co2"):
            st.markdown(
                """
                **Analyse :**
                Ce graphique illustre une relation positive entre la croissance de la population mondiale et les émissions de CO₂. 
                À mesure que la population augmente, les émissions de CO₂ suivent également une trajectoire ascendante, 
                ce qui suggère que l’augmentation de la population mondiale est un facteur clé dans la hausse des émissions. 
                Cela est lié à une plus forte demande en énergie, en transport et en infrastructures, qui sont souvent associées 
                à une augmentation des combustibles fossiles et, par conséquent, des émissions de CO₂.
                """
            )

        # Évolution du PIB mondial et des émissions de CO₂ (1880-2022)
        st.markdown(
            "### Évolution du PIB Mondial et des Émissions de CO₂ (1880-2022)")

        fig_pib_co2 = go.Figure()
        fig_pib_co2.add_trace(go.Scatter(
            x=final_data['Year'], y=final_data['GDP (billions)'], mode='lines', name='PIB', line=dict(color="green"), yaxis="y1"))
        fig_pib_co2.add_trace(go.Scatter(
            x=final_data['Year'], y=final_data['CO2 (kt)'], mode='lines', name='CO₂', line=dict(color="red"), yaxis="y2"))
        fig_pib_co2.update_layout(
            title="Évolution du PIB mondial et des émissions de CO₂ (1880-2022)",
            xaxis=dict(title='Année'),
            yaxis=dict(title='PIB (milliards)', titlefont=dict(
                color="green"), tickfont=dict(color="green")),
            yaxis2=dict(title='Émissions de CO₂ (kt)', titlefont=dict(
                color="red"), tickfont=dict(color="red"), overlaying='y', side='right')
        )
        st.plotly_chart(fig_pib_co2, use_container_width=True)

        # Bouton pour afficher les commentaires
        if st.checkbox("Je veux en savoir plus sur ce graphe", key="pib_co2"):
            st.markdown(
                """
                **Analyse :**
                Ce graphique montre l'évolution parallèle du PIB mondial (en vert) et des émissions de CO₂ (en rouge). 
                À mesure que le PIB augmente, les émissions de CO₂ augmentent également, ce qui indique un lien 
                entre la croissance économique et l’intensification de la pollution carbone. Les hausses conjointes 
                dans les deux courbes soulignent les défis que pose la croissance économique pour la durabilité environnementale, 
                et la nécessité de trouver des moyens d’augmenter le PIB sans accroître les émissions de CO₂.
                """
            )

        # Matrice de corrélation entre les variables
        st.markdown("### Matrice de Corrélation entre les Variables")

        corr_matrix = final_data.corr()
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Matrice de corrélation entre les variables")
        st.pyplot(fig_corr)

        # Bouton pour afficher les commentaires
        if st.checkbox("Je veux en savoir plus sur ce graphe", key="corr"):
            st.markdown(
                """
                **Analyse approfondie de la matrice de corrélation :**

                Cette matrice de corrélation présente les relations linéaires entre différentes variables importantes pour l'étude du changement climatique, notamment les températures, les émissions de CO₂, la population mondiale, et le PIB. Les valeurs de corrélation varient de -1 (corrélation négative parfaite) à 1 (corrélation positive parfaite). Voici une analyse des corrélations clés observées :

                - **Températures globales (Glob) et hémisphères (NHem, SHem) :** La température globale présente une corrélation très forte avec les températures de l'hémisphère nord (0.98) et de l'hémisphère sud (0.97). Cela montre que les augmentations de température sont globales, affectant les deux hémisphères de manière similaire, même si l'hémisphère nord est légèrement plus affecté, ce qui est souvent lié à une concentration plus élevée de population et d'activités industrielles.

                - **Population mondiale et émissions de CO₂ :** La population mondiale est fortement corrélée avec les émissions de CO₂ (0.94), indiquant que l'augmentation de la population contribue à l'augmentation des émissions. Avec une population croissante, la demande en énergie, transport, et ressources naturelles augmente, entraînant des émissions de CO₂ accrues.

                - **PIB (GDP) et émissions de CO₂ :** La corrélation entre le PIB et les émissions de CO₂ est également élevée (0.96). Cela souligne l'interconnexion entre la croissance économique et l'augmentation des émissions de gaz à effet de serre. En effet, de nombreuses économies sont encore dépendantes des combustibles fossiles, ce qui explique pourquoi une croissance du PIB entraîne généralement une hausse des émissions.

                - **Population et PIB :** La corrélation entre la population et le PIB (0.97) montre que les deux augmentent souvent conjointement. Une population plus importante peut favoriser la croissance économique en augmentant la main-d'œuvre et la consommation. Cependant, cette corrélation peut aussi indiquer que la croissance économique et l'industrialisation accompagnent l'augmentation de la population dans de nombreux pays.

                - **Année et variables climatiques :** Les fortes corrélations entre l'année et les autres variables, notamment les émissions de CO₂ (0.95) et la population (0.94), montrent que ces facteurs ont tous suivi une tendance à la hausse au fil des décennies. Cela est conforme à la tendance historique d'industrialisation et de croissance démographique depuis le début du 20e siècle.

                Cette matrice met en évidence les interconnexions entre la croissance démographique, le développement économique, et les émissions de gaz à effet de serre. Les corrélations fortes observées ici soulignent l'importance de prendre en compte plusieurs facteurs (population, PIB, émissions) dans les politiques de lutte contre le réchauffement climatique. Cela suggère aussi que réduire les émissions de CO₂ tout en maintenant une croissance économique sera un défi majeur pour les années à venir.
                """
            )

    except FileNotFoundError:
        st.error(
            "Le fichier final_data.csv est introuvable. Vérifiez le chemin et réessayez.")

##########################################################################
# Page Modèles et Prédictions
elif page == "Modèles et Prédictions":
    # Chargement des données pour la modélisation
    try:
        final_data_path = "Data/final_data.csv"
        final_data = pd.read_csv(final_data_path)
        st.write("Jeu de données final chargé avec succès.")
        
        # Calcul des taux de croissance annuels historiques pour définir les valeurs par défaut des curseurs
        gdp_growth_rate_default = final_data['GDP (billions)'].pct_change().mean() * 100
        population_growth_rate_default = final_data['Population (millions)'].pct_change().mean() * 100
        co2_growth_rate_default = final_data['CO2 (kt)'].pct_change().mean() * 100
    
    except FileNotFoundError:
        st.error("Le fichier final_data.csv est introuvable. Vérifiez le chemin et réessayez.")
    
    # Entraînement du modèle de régression linéaire final
    X = final_data[['Population (millions)', 'CO2 (kt)', 'GDP (billions)']]
    y = final_data['Glob']
    model_final = LinearRegression()
    model_final.fit(X, y)

    # Section 1 : Choix du modèle prédictif
    st.markdown("### Choix du modèle prédictif")
    models = [
        ("Régression Linéaire", LinearRegression(), ["CO2 (kt)"]),
        ("Régression Linéaire", LinearRegression(), ["CO2 (kt)", "GDP (billions)"]),
        ("Régression Linéaire", LinearRegression(), ["CO2 (kt)", "GDP (billions)", "Population (millions)"]),
        ("Régression Lasso", Lasso(alpha=0.1), ["CO2 (kt)"]),
        ("Régression Lasso", Lasso(alpha=0.1), ["CO2 (kt)", "GDP (billions)"]),
        ("Régression Lasso", Lasso(alpha=0.1), ["CO2 (kt)", "GDP (billions)", "Population (millions)"]),
        ("Régression Ridge", Ridge(alpha=1.0), ["CO2 (kt)"]),
        ("Régression Ridge", Ridge(alpha=1.0), ["CO2 (kt)", "GDP (billions)"]),
        ("Régression Ridge", Ridge(alpha=1.0), ["CO2 (kt)", "GDP (billions)", "Population (millions)"]),
        ("Forêt Aléatoire", RandomForestRegressor(n_estimators=100, random_state=42), ["CO2 (kt)"]),
        ("Forêt Aléatoire", RandomForestRegressor(n_estimators=100, random_state=42), ["CO2 (kt)", "GDP (billions)"]),
        ("Forêt Aléatoire", RandomForestRegressor(n_estimators=100, random_state=42), ["CO2 (kt)", "GDP (billions)", "Population (millions)"]),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42), ["CO2 (kt)"]),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42), ["CO2 (kt)", "GDP (billions)"]),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42), ["CO2 (kt)", "GDP (billions)", "Population (millions)"]),
        ("Réseau de Neurones", MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), ["CO2 (kt)"]),
        ("Réseau de Neurones", MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), ["CO2 (kt)", "GDP (billions)"]),
        ("Réseau de Neurones", MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), ["CO2 (kt)", "GDP (billions)", "Population (millions)"])
    ]

    results = []

    for model_name, model, features in models:
        X = final_data[features]
        y = final_data['Glob']
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        results.append({
            "Modèle": model_name,
            "Variables explicatives": " - ".join(features),
            "R²": f"{r2:.4f}",
            "MSE": f"{mse:.4f}"
        })

    results_df = pd.DataFrame(results)

    def highlight_row(row):
        color = 'background-color: green; color: white;' if row.name == 2 else ''
        return [color] * len(row)

    styled_df = results_df.style.apply(highlight_row, axis=1)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Tableau de synthèse des métriques des modèles")
        st.dataframe(styled_df)
    with col2:
        st.markdown("""
        ### Pourquoi avoir choisi la régression linéaire ?
        1. **Simplicité et interprétabilité** : les coefficients associés à chaque variable explicative permettent de comprendre clairement l'impact de chaque variable sur la température globale.
        2. **Bonne performance sur les données** : le R² de 0.9003 est satisfaisant pour une analyse prédictive.
        3. **Robustesse et généralisation** : contrairement à certains modèles non linéaires qui peuvent surajuster, la régression linéaire reste simple et robuste.
        """)

    # Définition des fonctions pour chaque scénario du GIEC
    def scenario_curve(coeffs, x):
        return coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]

    coeffs_ssp1_1_9 = np.polyfit([2020, 2050, 2100], [1.1, 1.3, 1.4], 2)
    coeffs_ssp1_2_6 = np.polyfit([2020, 2050, 2100], [1.1, 1.4, 1.8], 2)
    coeffs_ssp2_4_5 = np.polyfit([2020, 2050, 2100], [1.1, 1.8, 2.7], 2)
    coeffs_ssp3_7_0 = np.polyfit([2020, 2050, 2100], [1.1, 2.2, 3.6], 2)
    coeffs_ssp5_8_5 = np.polyfit([2020, 2050, 2100], [1.1, 2.4, 4.4], 2)

    scenario_functions = {
        "SSP1-1.9": lambda x: scenario_curve(coeffs_ssp1_1_9, x),
        "SSP1-2.6": lambda x: scenario_curve(coeffs_ssp1_2_6, x),
        "SSP2-4.5": lambda x: scenario_curve(coeffs_ssp2_4_5, x),
        "SSP3-7.0": lambda x: scenario_curve(coeffs_ssp3_7_0, x),
        "SSP5-8.5": lambda x: scenario_curve(coeffs_ssp5_8_5, x)
    }

    # Section 2 : Prédictions
    st.markdown("### Prédictions")

    # Boutons pour les options de réinitialisation et de corrélation
    col1, col2 = st.columns(2)
    with col1:
        reset_to_historical = st.button("Taux historiques")
    with col2:
        correlate_variables = st.checkbox("Corréler les taux")

    # Réinitialisation des valeurs des curseurs aux valeurs par défaut si le bouton est cliqué
    if reset_to_historical:
        gdp_growth_adjustment = gdp_growth_rate_default
        population_growth_adjustment = population_growth_rate_default
        co2_growth_adjustment = co2_growth_rate_default
    else:
        # Calcul des ratios de proportionnalité
        population_ratio = population_growth_rate_default / gdp_growth_rate_default
        co2_ratio = co2_growth_rate_default / gdp_growth_rate_default

        if correlate_variables:
            # Curseur principal pour ajuster le taux de croissance du PIB
            gdp_growth_adjustment = st.slider("Taux de croissance PIB (%)", -5.0, 5.0, gdp_growth_rate_default)

            # Calcul proportionnel pour les autres taux en gardant les ratios constants
            population_growth_adjustment = gdp_growth_adjustment * population_ratio
            co2_growth_adjustment = gdp_growth_adjustment * co2_ratio

            # Affichage des curseurs désactivés avec les valeurs calculées
            st.slider("Taux de croissance Population (%) (corrélé)", -5.0, 5.0, population_growth_adjustment, disabled=True)
            st.slider("Taux de croissance CO₂ (%) (corrélé)", -5.0, 5.0, co2_growth_adjustment, disabled=True)
        else:
            # Curseurs indépendants pour chaque taux de croissance
            gdp_growth_adjustment = st.slider("Taux de croissance PIB (%)", -5.0, 5.0, gdp_growth_rate_default)
            population_growth_adjustment = st.slider("Taux de croissance Population (%)", -5.0, 5.0, population_growth_rate_default)
            co2_growth_adjustment = st.slider("Taux de croissance CO₂ (%)", -5.0, 5.0, co2_growth_rate_default)

    # Calcul des projections des variables explicatives jusqu'en 2100 avec les taux ajustés
    years_projection = np.arange(2023, 2101)
    gdp_2022 = final_data['GDP (billions)'].values[-1]
    population_2022 = final_data['Population (millions)'].values[-1]
    co2_2022 = final_data['CO2 (kt)'].values[-1]

    gdp_projection = [gdp_2022 * ((1 + gdp_growth_adjustment / 100) ** (year - 2022)) for year in years_projection]
    population_projection = [population_2022 * ((1 + population_growth_adjustment / 100) ** (year - 2022)) for year in years_projection]
    co2_projection = [co2_2022 * ((1 + co2_growth_adjustment / 100) ** (year - 2022)) for year in years_projection]

    projection_df = pd.DataFrame({
        "Year": years_projection,
        "GDP (billions)": gdp_projection,
        "Population (millions)": population_projection,
        "CO2 (kt)": co2_projection
    })

    # Prédictions de température basées sur les projections
    X_projection = projection_df[['Population (millions)', 'CO2 (kt)', 'GDP (billions)']]
    temperature_predictions = model_final.predict(X_projection)
    projection_df['Predicted Temperature'] = temperature_predictions

    # Calcul de l'intervalle de confiance (exemple ±0.5°C autour de la projection)
    confidence_interval = 0.5
    projection_df['Upper Bound'] = projection_df['Predicted Temperature'] + confidence_interval
    projection_df['Lower Bound'] = projection_df['Predicted Temperature'] - confidence_interval

    # Affichage du graphique des prédictions de température avec option d'inclusion des scénarios du GIEC
    show_giec_scenarios = st.checkbox("Afficher les scénarios du GIEC")
    fig_temperature = go.Figure()

    fig_temperature.add_trace(go.Scatter(
        x=final_data['Year'], y=final_data['Glob'], mode='lines',
        name='Températures Observées (1880-2022)', line=dict(color='blue')
    ))
    fig_temperature.add_trace(go.Scatter(
        x=projection_df['Year'], y=projection_df['Predicted Temperature'], mode='lines',
        name='Températures Prédites (2023-2100)', line=dict(color='red', width=3)
    ))

    if not show_giec_scenarios:
        fig_temperature.add_trace(go.Scatter(
            x=np.concatenate([projection_df['Year'], projection_df['Year'][::-1]]),
            y=np.concatenate([projection_df['Upper Bound'], projection_df['Lower Bound'][::-1]]),
            fill='toself', fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=True, name='Intervalle de Confiance (±0.5°C)'
        ))

    if show_giec_scenarios:
        ssp_scenarios = {
            "SSP1-1.9": [scenario_functions["SSP1-1.9"](year) for year in years_projection],
            "SSP1-2.6": [scenario_functions["SSP1-2.6"](year) for year in years_projection],
            "SSP2-4.5": [scenario_functions["SSP2-4.5"](year) for year in years_projection],
            "SSP3-7.0": [scenario_functions["SSP3-7.0"](year) for year in years_projection],
            "SSP5-8.5": [scenario_functions["SSP5-8.5"](year) for year in years_projection]
        }
        colors = {"SSP1-1.9": "deepskyblue", "SSP1-2.6": "blue", "SSP2-4.5": "orange", "SSP3-7.0": "red", "SSP5-8.5": "darkred"}
        for scenario, values in ssp_scenarios.items():
            fig_temperature.add_trace(go.Scatter(
                x=years_projection, y=values, mode='lines',
                name=scenario, line=dict(color=colors[scenario], dash="dash")
            ))

    fig_temperature.update_layout(
        title='Prédictions de la Température Globale (2023-2100)',
        xaxis_title='Année',
        yaxis_title='Température Globale (°C)',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            itemwidth=50,
            traceorder="normal"
        ),
        width=700,
        height=500
    )

    # Graphique pour les projections des variables explicatives
    fig_projection = go.Figure()

    fig_projection.add_trace(go.Scatter(
        x=projection_df["Year"], y=projection_df["GDP (billions)"],
        mode='lines', name='PIB (milliards)', line=dict(color='blue'), yaxis='y1'
    ))
    fig_projection.add_trace(go.Scatter(
        x=projection_df["Year"], y=projection_df["Population (millions)"],
        mode='lines', name='Population (millions)', line=dict(color='green'), yaxis='y2'
    ))
    fig_projection.add_trace(go.Scatter(
        x=projection_df["Year"], y=projection_df["CO2 (kt)"],
        mode='lines', name='CO₂ (kt)', line=dict(color='red'), yaxis='y3'
    ))

    fig_projection.update_layout(
        title="Projection des Variables Explicatives jusqu'en 2100",
        xaxis=dict(title='Année'),
        yaxis=dict(title='PIB (milliards)', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Population (millions)', titlefont=dict(color='green'), tickfont=dict(color='green'),
                    overlaying='y', side='right'),
        yaxis3=dict(title='CO₂ (kt)', titlefont=dict(color='red'), tickfont=dict(color='red'),
                    anchor='free', overlaying='y', side='right', position=0.85),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        width=700, height=500
    )

    # Affichage des graphiques dans Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_projection, use_container_width=True)
    with col2:
        st.plotly_chart(fig_temperature, use_container_width=True)
        
    # Valeurs cibles des variables pour chaque scénario en 2100
    target_values_2100 = {
        "SSP1-1.9": {"GDP (billions)": 90000, "Population (millions)": 10000, "CO2 (kt)": 10000},
        "SSP1-2.6": {"GDP (billions)": 95000, "Population (millions)": 10500, "CO2 (kt)": 20000},
        "SSP2-4.5": {"GDP (billions)": 100000, "Population (millions)": 11000, "CO2 (kt)": 30000},
        "SSP3-7.0": {"GDP (billions)": 105000, "Population (millions)": 11500, "CO2 (kt)": 40000},
        "SSP5-8.5": {"GDP (billions)": 110000, "Population (millions)": 12000, "CO2 (kt)": 50000},
    }

    # Initial values for the variables
    initial_values = {
        "GDP (billions)": gdp_2022,
        "Population (millions)": population_2022,
        "CO2 (kt)": co2_2022
    }

    # Calculation of theoretical growth rates
    growth_rates = []
    for scenario, targets in target_values_2100.items():
        scenario_growth = {"Scénario": scenario}
        for variable, target_value in targets.items():
            initial_value = initial_values[variable]
            years = 2100 - 2022
            growth_rate = ((target_value / initial_value) ** (1 / years) - 1) * 100
            scenario_growth[variable] = f"{growth_rate:.2f}%"
        
        growth_rates.append(scenario_growth)

    # Create DataFrame for display
    growth_df = pd.DataFrame(growth_rates)

    # Display the growth rates table at the bottom of the "Modèles et Prédictions" page
    st.markdown("### Taux de Croissance Théoriques pour Atteindre les Valeurs des Scénarios du GIEC en 2100")
    st.dataframe(growth_df)