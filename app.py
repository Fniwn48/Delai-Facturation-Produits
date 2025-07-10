import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration pour performances maximales
pd.set_option("styler.render.max_elements", 100000)
st.set_page_config(
    page_title="Délais de Facturation des Produits",
    layout="wide",
    initial_sidebar_state="expanded"
)


regions_map = {
        # Métropole
        "01": "Auvergne-Rhône-Alpes",
        "02": "Hauts-de-France",
        "03": "Auvergne-Rhône-Alpes",
        "04": "Provence-Alpes-Côte d'Azur",
        "05": "Provence-Alpes-Côte d'Azur",
        "06": "Provence-Alpes-Côte d'Azur",
        "07": "Auvergne-Rhône-Alpes",
        "08": "Grand Est",
        "09": "Occitanie",
        "10": "Grand Est",
        "11": "Occitanie",
        "12": "Occitanie",
        "13": "Provence-Alpes-Côte d'Azur",
        "14": "Normandie",
        "15": "Auvergne-Rhône-Alpes",
        "16": "Nouvelle-Aquitaine",
        "17": "Nouvelle-Aquitaine",
        "18": "Centre-Val de Loire",
        "19": "Nouvelle-Aquitaine",
        "21": "Bourgogne-Franche-Comté",
        "22": "Bretagne",
        "23": "Nouvelle-Aquitaine",
        "24": "Nouvelle-Aquitaine",
        "25": "Bourgogne-Franche-Comté",
        "26": "Auvergne-Rhône-Alpes",
        "27": "Normandie",
        "28": "Centre-Val de Loire",
        "29": "Bretagne",
        "30": "Occitanie",
        "31": "Occitanie",
        "32": "Occitanie",
        "33": "Nouvelle-Aquitaine",
        "34": "Occitanie",
        "35": "Bretagne",
        "36": "Centre-Val de Loire",
        "37": "Centre-Val de Loire",
        "38": "Auvergne-Rhône-Alpes",
        "39": "Bourgogne-Franche-Comté",
        "40": "Nouvelle-Aquitaine",
        "41": "Centre-Val de Loire",
        "42": "Auvergne-Rhône-Alpes",
        "43": "Auvergne-Rhône-Alpes",
        "44": "Pays de la Loire",
        "45": "Centre-Val de Loire",
        "46": "Occitanie",
        "47": "Nouvelle-Aquitaine",
        "48": "Occitanie",
        "49": "Pays de la Loire",
        "50": "Normandie",
        "51": "Grand Est",
        "52": "Grand Est",
        "53": "Pays de la Loire",
        "54": "Grand Est",
        "55": "Grand Est",
        "56": "Bretagne",
        "57": "Grand Est",
        "58": "Bourgogne-Franche-Comté",
        "59": "Hauts-de-France",
        "60": "Hauts-de-France",
        "61": "Normandie",
        "62": "Hauts-de-France",
        "63": "Auvergne-Rhône-Alpes",
        "64": "Nouvelle-Aquitaine",
        "65": "Occitanie",
        "66": "Occitanie",
        "67": "Grand Est",
        "68": "Grand Est",
        "69": "Auvergne-Rhône-Alpes",
        "70": "Bourgogne-Franche-Comté",
        "71": "Bourgogne-Franche-Comté",
        "72": "Pays de la Loire",
        "73": "Auvergne-Rhône-Alpes",
        "74": "Auvergne-Rhône-Alpes",
        "75": "Île-de-France",
        "76": "Normandie",
        "77": "Île-de-France",
        "78": "Île-de-France",
        "79": "Nouvelle-Aquitaine",
        "80": "Hauts-de-France",
        "81": "Occitanie",
        "82": "Occitanie",
        "83": "Provence-Alpes-Côte d'Azur",
        "84": "Provence-Alpes-Côte d'Azur",
        "85": "Pays de la Loire",
        "86": "Nouvelle-Aquitaine",
        "87": "Nouvelle-Aquitaine",
        "88": "Grand Est",
        "89": "Bourgogne-Franche-Comté",
        "90": "Bourgogne-Franche-Comté",
        "91": "Île-de-France",
        "92": "Île-de-France",
        "93": "Île-de-France",
        "94": "Île-de-France",
        "95": "Île-de-France"
}

# Cache ultra-agressif pour toutes les fonctions lourdes
@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def get_french_region_vectorized(postal_codes):
    """Version vectorisée ultra-rapide pour les régions - CORRIGÉE pour toutes les régions administratives"""
    # Normalisation des codes postaux
    postal_codes_str = postal_codes.astype(str).str.strip()
    
    # Remplacer les valeurs vides/nulles par une valeur par défaut
    postal_codes_str = postal_codes_str.replace(['', 'nan', 'None', 'NaN'], '00000')
    
    # Fonction pour extraire le code département de manière robuste
    def extract_dept_code(postal_code):
        if len(postal_code) < 2:
            return "00"
        
        # Pour les codes postaux de 5 chiffres (métropole standard)
        if len(postal_code) == 5 and postal_code.isdigit():
            if postal_code.startswith('20'):
                # Gestion spéciale de la Corse
                if postal_code[:3] in ['200', '201']:
                    return "2A"  # Corse-du-Sud
                elif postal_code[:3] in ['202', '206']:
                    return "2B"  # Haute-Corse
                else:
                    return "20"  # Code générique Corse
            else:
                return postal_code[:2]
        
        # Pour les codes postaux de 4 chiffres (ajouter un 0 au début)
        elif len(postal_code) == 4 and postal_code.isdigit():
            return "0" + postal_code[0]
        
        # Pour les codes postaux d'outre-mer (3 chiffres + 2 chiffres)
        elif len(postal_code) == 5 and postal_code[:3] in ['971', '972', '973', '974', '975', '976', '977', '978']:
            return postal_code[:3]
        
        # Codes spéciaux plus longs (TAAF, Wallis, Polynésie, Nouvelle-Calédonie)
        elif len(postal_code) >= 5:
            if postal_code.startswith('984'):
                return "984"
            elif postal_code.startswith('986'):
                return "986"  
            elif postal_code.startswith('987'):
                return "987"
            elif postal_code.startswith('988'):
                return "988"
            elif postal_code.startswith('98'):
                return "98"  # Monaco
            else:
                return postal_code[:2]
        
        # Pour les codes déjà au format département (2 caractères)
        elif len(postal_code) == 2:
            return postal_code.upper()
        
        # Cas particuliers et codes spéciaux
        elif postal_code.upper() in ['2A', '2B']:
            return postal_code.upper()
        
        # Codes postaux étrangers ou militaires
        elif postal_code.startswith('00') or postal_code.startswith('99'):
            return postal_code[:2]
        
        # Cas par défaut
        else:
            return postal_code[:2] if len(postal_code) >= 2 else "00"
    
    # Application vectorielle de l'extraction
    dept_codes = postal_codes_str.apply(extract_dept_code)
    
    # Mapping vers les régions avec gestion des codes non trouvés
    regions = dept_codes.map(regions_map)
    
    # Remplacement des valeurs non trouvées
    regions = regions.fillna("Autre région")
    
    return regions

def clean_numeric_column(series):
    """Nettoie une colonne pour la convertir en numérique"""
    # Convertir en string d'abord
    series = series.astype(str)
    
    # Remplacer les NaN, None, 'nan' par 0
    series = series.replace(['nan', 'None', 'NaN', ''], '0')
    
    # Supprimer les symboles de pourcentage et autres caractères
    series = series.str.replace('%', '', regex=False)
    series = series.str.replace(',', '.', regex=False)  # Remplacer virgules par points
    series = series.str.replace(' ', '', regex=False)   # Supprimer espaces
    series = series.str.replace('€', '', regex=False)   # Supprimer symboles monétaires
    
    # Convertir en numérique
    series = pd.to_numeric(series, errors='coerce')
    
    # Remplacer les NaN par 0
    series = series.fillna(0)
    
    return series

@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_vc_file(vc_file_data):
    """Charge le fichier VC et retourne la liste des materials VC"""
    try:
        vc_df = pd.read_excel(vc_file_data, engine='openpyxl')
        
        # Vérifier si la colonne Material existe
        if 'Material' not in vc_df.columns:
            return None, "La colonne 'Material' n'est pas trouvée dans le fichier VC"
        
        # Récupérer la liste unique des materials VC
        vc_materials = vc_df['Material'].dropna().unique().tolist()
        
        return vc_materials, f"✅ {len(vc_materials)} materials VC chargés"
        
    except Exception as e:
        return None, f"Erreur lors du chargement du fichier VC: {str(e)}"

@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_and_process_data_ultra_fast(file_data, vc_materials=None):
    """Version ultra-optimisée du chargement de données avec gestion des erreurs de conversion et ajout Type VC"""
    try:
        # Lecture sans spécifier les types pour les colonnes numériques problématiques
        dtype_dict = {
            'Sales Document #': 'string',
            'Material Y#': 'string', 
            'Material Desc': 'string',
            'Material MRP Controller': 'string',
            'Product Line Desc': 'string',
            'ShipTo #': 'string',
            'ShipTo Name': 'string',
            'SoldTo City': 'string',
            'SoldTo Postal Code': 'string',
            'SoldTo Country': 'string'
            # On ne spécifie pas les types pour les colonnes numériques
        }
        
        df = pd.read_excel(file_data, dtype=dtype_dict, engine='openpyxl')
        
        # Vérification colonnes - optimisée
        required_columns = [
            'Posting Date', 'Customer Purchase Order Date', 'Sales Document #',
            'Material Y#', 'Material Desc', 'Material MRP Controller',
            'Product Line Desc', 'ShipTo #', 'ShipTo Name', 'SoldTo City', 'SoldTo Postal Code', 'SoldTo Country',
            'Customer Sales', 'Customer Margin $', 'Customer Margin %'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Colonnes manquantes: {', '.join(missing_columns)}"
        
        # Nettoyage des colonnes numériques AVANT toute autre opération
        df['Customer Sales'] = clean_numeric_column(df['Customer Sales'])
        df['Customer Margin $'] = clean_numeric_column(df['Customer Margin $'])
        df['Customer Margin %'] = clean_numeric_column(df['Customer Margin %'])
        
        # Conversion dates ultra-rapide
        df['Posting Date'] = pd.to_datetime(df['Posting Date'], errors='coerce', infer_datetime_format=True)
        df['Customer Purchase Order Date'] = pd.to_datetime(df['Customer Purchase Order Date'], errors='coerce', infer_datetime_format=True)
        
        # Calcul des délais UNIQUEMENT pour les lignes avec les deux dates valides
        mask_dates_valides = df['Posting Date'].notna() & df['Customer Purchase Order Date'].notna()
        df['Écart en jours'] = np.nan  # Initialiser avec NaN
        df.loc[mask_dates_valides, 'Écart en jours'] = (
            df.loc[mask_dates_valides, 'Posting Date'] - 
            df.loc[mask_dates_valides, 'Customer Purchase Order Date']
        ).dt.days
        
        # Calculs vectorisés ULTRA-RAPIDES
        df['Année Fiscale'] = np.where(df['Posting Date'].dt.month >= 8, 
                                       df['Posting Date'].dt.year + 1, 
                                       df['Posting Date'].dt.year)
        
        # Catégorie Make/Buy - vectorisé ultra-rapide
        make_controllers = {'M01', 'M04', 'M06', 'MDM', 'P0E', 'P0F', 'P0N', 'PFA', 'P0G', 'PNP', 'PPD', 'PPO', 'PSD','M70'}
        # Catégorie Make/Buy - avec exception pour Y4950100
        df['Catégorie Produit'] = np.where(
            (df['Material MRP Controller'].str.strip().isin(make_controllers)) & (df['Material Y#'] != 'Y4950100'), 
            'Make', 
            'Buy'
        )        

        # AJOUT DE LA COLONNE TYPE VC
        if vc_materials is not None:
            # Convertir vc_materials en set pour des recherches plus rapides
            vc_materials_set = set(vc_materials)
            
            def categorize_material(x):
                # Gérer les valeurs NaN/None
                if pd.isna(x) or x is None:
                    return "Standard"
                # Convertir en string pour la comparaison
                x_str = str(x).strip()
                if x_str == "Y5010646":
                    return "Install"
                elif x_str in vc_materials_set:
                    return "VC"
                else:
                    return "Standard"
            
            df['Type VC'] = df['Material Y#'].apply(categorize_material)
        else:
            # Si pas de fichier VC, tout est Standard sauf Y5010646
            df['Type VC'] = df['Material Y#'].apply(
                lambda x: "Install" if pd.notna(x) and str(x).strip() == "Y5010646" else "Standard"
            )

        # Identifiants - utiliser seulement Material Y# pour les produits
        df['Produit Unique'] = df['Material Y#'].astype(str)
        df['Client Unique'] = df['ShipTo #'].astype(str) + '_' + df['ShipTo Name'].astype(str)
        
        # Région - fonction vectorisée
        df['Région'] = get_french_region_vectorized(df['SoldTo Postal Code'])
        
        return df, "Success"
        
    except Exception as e:
        return None, f"Erreur lors du traitement: {str(e)}"

def format_dataframe_for_display(df, df_type="standard"):
    """Formate les DataFrames pour l'affichage avec les bonnes unités"""
    df_formatted = df.copy()
    
    # Formatage selon le type de DataFrame
    if df_type == "product":
        # Pour les produits : Customer Sales, Customer Margin $, % Marge
        df_formatted['Customer Sales'] = df_formatted['Customer Sales'].apply(lambda x: f"{x:,.2f} €")
        df_formatted['Customer Margin $'] = df_formatted['Customer Margin $'].apply(lambda x: f"{x:,.2f} €")
        df_formatted['% Marge'] = df_formatted['% Marge'].apply(lambda x: f"{x:.2f} %")
        df_formatted['Écart en jours'] = df_formatted['Écart en jours'].apply(lambda x: f"{x:.1f} j" if pd.notna(x) else "N/A")
        df_formatted = df_formatted.rename(columns={
            'Material Y#': 'Material',
            'Customer Sales': 'CA',
            'Customer Margin $': 'Marge €',
            'Écart en jours': 'Délai Moy.'
        })
            
    elif df_type == "client":
        # Pour les clients
        df_formatted['Customer Sales'] = df_formatted['Customer Sales'].apply(lambda x: f"{x:,.2f} €")
        df_formatted['Customer Margin $'] = df_formatted['Customer Margin $'].apply(lambda x: f"{x:,.2f} €")
        df_formatted['% Marge'] = df_formatted['% Marge'].apply(lambda x: f"{x:.2f} %")
        df_formatted['Écart en jours'] = df_formatted['Écart en jours'].apply(lambda x: f"{x:.1f} j" if pd.notna(x) else "N/A")
        
        # Renommer les colonnes
        df_formatted = df_formatted.rename(columns={
            'ShipTo #': 'N° Client',
            'ShipTo Name': 'Nom Client',
            'SoldTo City': 'Ville',
            'SoldTo Postal Code': 'CP',
            'Sales Document #': 'Nb Commandes',
            'Produit Unique': 'Nb Produits',
            'Material Y#': 'Nb Lignes',
            'Customer Sales': 'CA',
            'Customer Margin $': 'Marge €',
            '% Marge': 'Marge %',
            'Écart en jours': 'Délai Moy.'
        })
        
    elif df_type == "city":
        # Pour les villes
        df_formatted['Customer Sales'] = df_formatted['Customer Sales'].apply(lambda x: f"{x:,.2f} €")
        df_formatted['Customer Margin $'] = df_formatted['Customer Margin $'].apply(lambda x: f"{x:,.2f} €")
        df_formatted['Marge %'] = df_formatted['Marge %'].apply(lambda x: f"{x:.2f} %")
        df_formatted['Écart en jours'] = df_formatted['Écart en jours'].apply(lambda x: f"{x:.1f} j" if pd.notna(x) else "N/A")
        
        # Renommer les colonnes
        df_formatted = df_formatted.rename(columns={
            'SoldTo City': 'Ville',
            'SoldTo Postal Code': 'CP',
            'Client Unique': 'Nb Clients',
            'Sales Document #': 'Nb Commandes',
            'Produit Unique': 'Nb Produits',
            'Customer Sales': 'CA',
            'Customer Margin $': 'Marge €',
            'Marge %': 'Marge %',
            'Écart en jours': 'Délai Moy.'
        })
    
    return df_formatted

@st.cache_data(ttl=1800, max_entries=20, show_spinner=False)
def create_all_analyses_batch(df):
    """Calcule TOUTES les analyses en une seule fois - ULTRA OPTIMISÉ"""
    if len(df) == 0:
        return {}, {}, {}, {}, {}
    
    # Métriques globales - vectorisé
    # Pour les délais, on ne prend que les lignes avec des dates valides
    delais_valides = df['Écart en jours'].dropna()
    
    metrics = {
        'Moyenne jours facturation': delais_valides.mean() if len(delais_valides) > 0 else 0,
        'Produits différents': df['Produit Unique'].nunique(),
        'Nombre total lignes': len(df),
        'Commandes différentes': df['Sales Document #'].nunique(),
        'Clients différents': df['Client Unique'].nunique(),
        'CA total (€)': df['Customer Sales'].sum(),
        'Marge totale (€)': df['Customer Margin $'].sum()
    }
    
    # Analyse produits - groupby optimisé avec délais min/max et filtrage CA positif
    # Utiliser seulement Material Y# pour identifier les produits
    product_analysis = df.groupby(['Material Y#', 'Material Desc']).agg({
        'Customer Sales': 'sum',
        'Customer Margin $': 'sum',
        'Sales Document #': 'nunique',
        'Écart en jours': ['mean', 'min', 'max']  # Liste de fonctions
    }).reset_index()

    # Aplatir les colonnes multi-index
    product_analysis.columns = ['Material Y#', 'Material Desc', 
                            'Customer Sales', 'Customer Margin $', 'Nb Commandes',
                            'Écart en jours', 'Délai Min', 'Délai Max']

    # Supprimer les produits avec CA négatif
    product_analysis = product_analysis[product_analysis['Customer Sales'] > 0]
    
    product_analysis['% Marge'] = np.where(
        product_analysis['Customer Sales'] > 0,
        (product_analysis['Customer Margin $'] / product_analysis['Customer Sales'] * 100),
        0
    )
    product_analysis = product_analysis.sort_values('Customer Sales', ascending=False)
    
    # Analyse clients - groupby optimisé avec rename
    client_analysis = df.groupby(['ShipTo #', 'ShipTo Name', 'SoldTo City', 'SoldTo Postal Code']).agg({
        'Sales Document #': 'nunique',
        'Produit Unique': 'nunique',
        'Material Y#': 'count',
        'Customer Sales': 'sum',
        'Customer Margin $': 'sum',
        'Écart en jours': 'mean'
    }).reset_index()
    
    client_analysis['% Marge'] = np.where(
        client_analysis['Customer Sales'] > 0,
        (client_analysis['Customer Margin $'] / client_analysis['Customer Sales'] * 100),
        0
    )
    client_analysis = client_analysis.sort_values('Customer Sales', ascending=False)
    
    # Analyse par ville - groupby optimisé
    city_analysis = df.groupby(['SoldTo City', 'SoldTo Postal Code', 'Région']).agg({
        'Client Unique': 'nunique',
        'Sales Document #': 'nunique',
        'Produit Unique': 'nunique',
        'Customer Sales': 'sum',
        'Customer Margin $': 'sum',
        'Écart en jours': 'mean'
    }).reset_index()
    
    city_analysis['Marge %'] = np.where(
        city_analysis['Customer Sales'] > 0,
        (city_analysis['Customer Margin $'] / city_analysis['Customer Sales'] * 100),
        0
    )
    city_analysis = city_analysis.sort_values('Customer Sales', ascending=False)

    
    return metrics, product_analysis, client_analysis, city_analysis

@st.cache_data(ttl=1800, max_entries=10, show_spinner=False)
def apply_filters_ultra_fast(df, fiscal_year, period_range, category, product_line, type_vc_list):
    """Application ultra-rapide des filtres avec masques vectorisés - MODIFIÉ pour multiselect Type VC"""
    if len(df) == 0:
        return df
    
    # Masque année fiscale - vectorisé
    mask = df['Année Fiscale'] == fiscal_year
    
    # Masque période - vectorisé
    if period_range != (8, 19):
        start_month = period_range[0] if period_range[0] <= 12 else period_range[0] - 12
        end_month = period_range[1] if period_range[1] <= 12 else period_range[1] - 12
        
        if period_range[0] <= 12 and period_range[1] <= 12:
            mask &= (df['Posting Date'].dt.month >= start_month) & (df['Posting Date'].dt.month <= end_month)
        elif period_range[0] > 12 and period_range[1] > 12:
            mask &= (df['Posting Date'].dt.month >= start_month) & (df['Posting Date'].dt.month <= end_month)
        else:
            mask &= (df['Posting Date'].dt.month >= start_month) | (df['Posting Date'].dt.month <= end_month)
    
    # Masques catégorie et gamme - vectorisés
    if category != 'Tous':
        mask &= df['Catégorie Produit'] == category
    
    if product_line != 'Tous':
        mask &= df['Product Line Desc'].astype(str) == product_line
    
    # AJOUT DU FILTRE TYPE VC - MULTISELECT
    if type_vc_list and len(type_vc_list) > 0:  # Si une sélection existe
        mask &= df['Type VC'].isin(type_vc_list)
    
    return df[mask].copy()

@st.cache_data(ttl=1800, max_entries=5, show_spinner=False)
def create_visualizations_batch(df):
    """Crée toutes les visualisations en batch - ULTRA OPTIMISÉ"""
    if len(df) == 0:
        return None, None
    
    # Évolution mensuelle - optimisée
    monthly_data = df.groupby(df['Posting Date'].dt.to_period('M'))['Customer Sales'].sum()
    fig_monthly = px.line(x=monthly_data.index.astype(str), y=monthly_data.values, title="📈 Évolution CA")
    fig_monthly.update_traces(line_color='#6c5ce7', line_width=3)
    fig_monthly.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
    
    # Répartition Make/Buy - optimisée
    category_data = df.groupby('Catégorie Produit')['Customer Sales'].sum()
    fig_category = px.pie(values=category_data.values, names=category_data.index, title="🏭 Make vs Buy")
    
    return fig_monthly, fig_category

@st.cache_data(ttl=1800, max_entries=5, show_spinner=False)
def create_client_visualizations(df):
    """Crée les visualisations spécifiques aux clients - CORRIGÉ pour toutes les régions"""
    if len(df) == 0:
        return None, None
    
    # Évolution mensuelle du nombre de clients - optimisée
    monthly_clients = df.groupby(df['Posting Date'].dt.to_period('M'))['Client Unique'].nunique()
    fig_monthly = px.line(x=monthly_clients.index.astype(str), y=monthly_clients.values, title="📈 Évolution Nombre de Clients")
    fig_monthly.update_traces(line_color='#6c5ce7', line_width=3)
    fig_monthly.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
    
    # Distribution des clients par RÉGION - CORRIGÉ pour pourcentage de clients
    region_client_counts = df.groupby('Région')['Client Unique'].nunique()  # Nombre de clients uniques par région
    total_unique_clients = df['Client Unique'].nunique()  # Total de clients uniques
    region_percentages = (region_client_counts / total_unique_clients * 100).round(1)  # Pourcentage de clients par région

    # Créer les labels avec région, nombre de clients et pourcentage de clients
    labels = [f"{region}<br>{count} clients ({pct}%)" 
             for region, count, pct in zip(region_client_counts.index, region_client_counts.values, region_percentages.values)]

    fig_category = px.pie(
        values=region_client_counts.values, 
        names=labels, 
        title="🗺️ Distribution Clients par Région"
    )
    fig_category.update_layout(height=400)
        
    return fig_monthly, fig_category

# CSS ultra-compact
st.markdown("""<style>
.main-header{background:linear-gradient(135deg,#6c5ce7 0%,#a29bfe 100%);padding:1rem;border-radius:8px;color:white;text-align:center;margin-bottom:1rem}
.metric-card{background:#f8f9fa;padding:0.5rem;border-radius:4px;border-left:3px solid #6c5ce7;margin:0.2rem 0;height:60px;display:flex;flex-direction:column;justify-content:center;box-shadow:0 1px 3px rgba(0,0,0,0.1)}
.metric-card h3{font-size:0.8rem;color:#6c757d;margin:0;font-weight:700;text-transform:uppercase}
.metric-card h2{font-size:0.9rem;color:#2d3436;margin:0;font-weight:600}
</style>""", unsafe_allow_html=True)

def main():
    # Interface ultra-minimaliste
    st.markdown('<div class="main-header"><h4>📊 Analyse des performances des produits</h></div>', unsafe_allow_html=True)
    
    # État de session pour stocker les fichiers
    if 'main_file_loaded' not in st.session_state:
        st.session_state.main_file_loaded = False
    if 'vc_file_loaded' not in st.session_state:
        st.session_state.vc_file_loaded = False
    if 'vc_materials' not in st.session_state:
        st.session_state.vc_materials = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if not st.session_state.main_file_loaded or not st.session_state.vc_file_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###### 📁 Fichier délais de facturation")
            uploaded_file = st.file_uploader("Sélectionnez le fichier Excel principal", type=['xlsx', 'xls'], key="main_file")
            
        with col2:
            st.markdown("###### 📁 Fichier VC")
            vc_file = st.file_uploader("Sélectionnez le fichier VC (optionnel)", type=['xlsx', 'xls'], key="vc_file")
        
        # Bouton pour charger les fichiers
        col1, col2, col3 = st.columns([4, 1, 4])
        with col2:
            if st.button("🚀 Charger les fichiers", type="primary"):
                if uploaded_file is not None:
                    # Charger d'abord le fichier VC si présent
                    if vc_file is not None:
                        with st.spinner("🔄 Chargement du fichier VC..."):
                            vc_materials, vc_status = load_vc_file(vc_file)
                            if vc_materials is not None:
                                st.session_state.vc_materials = vc_materials
                                st.session_state.vc_file_loaded = True
                                st.success(vc_status)
                            else:
                                st.error(vc_status)
                                return
                    else:
                        st.session_state.vc_materials = None
                        st.session_state.vc_file_loaded = True
                    
                    # Charger le fichier principal
                    with st.spinner("🔄 Chargement et nettoyage des données principales..."):
                        df, status = load_and_process_data_ultra_fast(uploaded_file, st.session_state.vc_materials)
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.main_file_loaded = True
                            st.success(f"✅ {len(df):,} lignes chargées avec succès!")
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur: {status}")
                            return
                else:
                    st.warning("⚠️ Veuillez sélectionner au moins le fichier principal")
        
        # Si les fichiers ne sont pas encore chargés, arrêter ici
        return
    
    # Si les fichiers sont chargés, utiliser les données du session state
    df = st.session_state.df
    
    # Bouton pour retourner à l'importation
    if st.sidebar.button("🏠 Recommencer", type="primary"):
        st.session_state.main_file_loaded = False
        st.session_state.vc_file_loaded = False
        st.session_state.vc_materials = None
        st.session_state.df = None
        st.rerun()
    
    # Filtres optimisés
    st.sidebar.markdown("#### Filtres")
    
    # Filtrer les années valides et les convertir en int
    available_years = df['Année Fiscale'].dropna().unique()
    available_years = sorted([int(year) for year in available_years if pd.notna(year)])
    fiscal_year = st.sidebar.selectbox("Année Fiscale", available_years)
    
    period_range = st.sidebar.select_slider(
        "Période", options=list(range(8, 20)), value=(8, 19),
        format_func=lambda x: ['Août','Sep','Oct','Nov','Déc','Jan','Fév','Mar','Avr','Mai','Jun','Jul'][x-8]
    )
    
    # Filtres interdépendants Catégorie/Gamme
    # D'abord, on filtre par année et période pour avoir les bonnes données
    mask_year_period = (df['Année Fiscale'] == fiscal_year)
    if period_range != (8, 19):
        start_month = period_range[0] if period_range[0] <= 12 else period_range[0] - 12
        end_month = period_range[1] if period_range[1] <= 12 else period_range[1] - 12
        
        if period_range[0] <= 12 and period_range[1] <= 12:
            mask_year_period &= (df['Posting Date'].dt.month >= start_month) & (df['Posting Date'].dt.month <= end_month)
        elif period_range[0] > 12 and period_range[1] > 12:
            mask_year_period &= (df['Posting Date'].dt.month >= start_month) & (df['Posting Date'].dt.month <= end_month)
        else:
            mask_year_period &= (df['Posting Date'].dt.month >= start_month) | (df['Posting Date'].dt.month <= end_month)
    
    df_filtered_year_period = df[mask_year_period]
    
    # Obtenir toutes les catégories et gammes disponibles
    all_categories = ['Tous'] + sorted(df_filtered_year_period['Catégorie Produit'].unique())
    all_product_lines = ['Tous'] + sorted(df_filtered_year_period['Product Line Desc'].dropna().astype(str).unique())
    
    # Sélection de la catégorie
    category = st.sidebar.selectbox("Catégorie", all_categories, key="category_select")
    
    # Si une catégorie est sélectionnée, filtrer les gammes disponibles
    if category != 'Tous':
        available_product_lines = df_filtered_year_period[df_filtered_year_period['Catégorie Produit'] == category]['Product Line Desc'].dropna().astype(str).unique()
        product_lines = ['Tous'] + sorted(available_product_lines)
    else:
        product_lines = all_product_lines
    
    # Sélection de la gamme
    product_line = st.sidebar.selectbox("Gamme", product_lines, key="product_line_select")
    
    # Si une gamme est sélectionnée et catégorie est "Tous", mettre à jour la catégorie
    if product_line != 'Tous' and category == 'Tous':
        available_categories = df_filtered_year_period[df_filtered_year_period['Product Line Desc'].astype(str) == product_line]['Catégorie Produit'].unique()
        if len(available_categories) == 1:
            category = available_categories[0]
            st.sidebar.info(f"Catégorie automatiquement définie : {category}")
    
    # AJOUT DU FILTRE TYPE VC - MULTISELECT
    # Obtenir les types VC disponibles après les autres filtres
    mask_filters = mask_year_period
    if category != 'Tous':
        mask_filters &= (df['Catégorie Produit'] == category)
    if product_line != 'Tous':
        mask_filters &= (df['Product Line Desc'].astype(str) == product_line)
    
    available_type_vc = sorted(df[mask_filters]['Type VC'].unique())
    type_vc_selected = st.sidebar.multiselect(
        "Type VC", 
        options=available_type_vc,
        default=available_type_vc,  # Par défaut, tout est sélectionné
        key="type_vc_multiselect"
    )
    
    # Application filtres ultra-rapide avec Type VC
    filtered_df = apply_filters_ultra_fast(df, fiscal_year, period_range, category, product_line, type_vc_selected)
    
    if len(filtered_df) == 0:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés")
        return
    
    # Calculs batch ultra-rapides
    with st.spinner("⚡ Calcul express..."):
        results = create_all_analyses_batch(filtered_df)
        if len(results) == 4:
            metrics, product_df, client_df, city_df = results
        else:
            st.error("Erreur lors des calculs")
            return
    
    # Métriques ultra-compactes
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    cols = [col1, col2, col3, col4, col5, col6]
    metric_data = [
        ("⏱️ Moy.Jours", f"{metrics['Moyenne jours facturation']:.1f}"),
        ("📦 Produits", f"{metrics['Produits différents']:,}"),
        ("📋 Lignes", f"{metrics['Nombre total lignes']:,}"),
        ("🛒 Commandes", f"{metrics['Commandes différentes']:,}"),
        ("👥 Clients", f"{metrics['Clients différents']:,}"),
        ("💰 CA", f"{metrics['CA total (€)']:,.2f}€")
    ]
    
    for i, (title, value) in enumerate(metric_data):
        with cols[i]:
            st.markdown(f'<div class="metric-card"><h3>{title}</h3><h2>{value}</h2></div>', unsafe_allow_html=True)
    
    # Tableaux ultra-optimisés avec formatage
    tab1, tab2, tab3 = st.tabs(["📦 Produits", "👥 Clients", "🏘️ Villes"])

    with tab1:
        if len(product_df) > 0:
            # AJOUT DE LA COLONNE TYPE VC DANS L'ANALYSE PRODUITS
            # Merge avec les données filtrées pour obtenir le Type VC
            product_df_with_type = product_df.merge(
                filtered_df[['Material Y#', 'Type VC']].drop_duplicates(), 
                on='Material Y#', 
                how='left'
            )
            
            # Réorganiser les colonnes pour mettre Type VC après Material Desc
            cols_order = ['Material Y#', 'Material Desc', 'Type VC', 'Customer Sales', 'Customer Margin $', 
                         '% Marge', 'Nb Commandes', 'Écart en jours', 'Délai Min', 'Délai Max']
            product_df_with_type = product_df_with_type[cols_order]
            
            # Afficher TOUS les produits et formater l'affichage
            display_df = format_dataframe_for_display(product_df_with_type, "product")
            st.dataframe(display_df, height=400, use_container_width=True, hide_index=True)
                # Graphiques additionnels
            st.markdown("#### 📊 Analyses Complémentaires")
            col1, col2 = st.columns(2)
            
            # Visualisations
            with st.spinner("📊 Génération des graphiques..."):
                figs = create_visualizations_batch(filtered_df)
                if figs and len(figs) == 2:
                    fig_monthly, fig_category = figs
                    
                    with col1:
                        st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(fig_category, use_container_width=True)
            
            # Analyse détaillée par gamme de produit
            if product_line == 'Tous':
                st.markdown("#### 🎯 Gammes de Produits")
                product_line_analysis = filtered_df.groupby('Product Line Desc').agg({
                    'Customer Sales': 'sum',
                    'Customer Margin $': 'sum',
                    'Écart en jours': 'mean',
                    'Produit Unique': 'nunique'
                }).reset_index()
                
                product_line_analysis['Marge %'] = np.where(
                    product_line_analysis['Customer Sales'] > 0,
                    (product_line_analysis['Customer Margin $'] / product_line_analysis['Customer Sales'] * 100),
                    0
                )
                
                product_line_analysis = product_line_analysis.sort_values('Customer Sales', ascending=False)

                display_df = product_line_analysis.copy()
                display_df['Customer Sales'] = display_df['Customer Sales'].apply(lambda x: f"{x:,.2f} €")
                display_df['Customer Margin $'] = display_df['Customer Margin $'].apply(lambda x: f"{x:,.2f} €")
                display_df['Marge %'] = display_df['Marge %'].apply(lambda x: f"{x:.2f} %")
                display_df['Écart en jours'] = display_df['Écart en jours'].apply(lambda x: f"{x:.1f} j" if pd.notna(x) else "N/A")
                
                # Renommer les colonnes pour l'affichage
                display_df = display_df.rename(columns={
                    'Product Line Desc': 'Gamme de Produit',
                    'Customer Sales': 'CA',
                    'Customer Margin $': 'Marge €',
                    'Marge %': 'Marge %',
                    'Écart en jours': 'Délai Moy.',
                    'Produit Unique': 'Nb Produits'
                })
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
            # AJOUT : Analyse par Type VC si le filtre est sur "Tous"
            if len(type_vc_selected) == len(available_type_vc):
                st.markdown("#### 🏷️ Analyse par Type VC")
                type_vc_analysis = filtered_df.groupby('Type VC').agg({
                    'Customer Sales': 'sum',
                    'Customer Margin $': 'sum',
                    'Écart en jours': 'mean',
                    'Produit Unique': 'nunique',
                    'Client Unique': 'nunique'
                }).reset_index()
                
                type_vc_analysis['Marge %'] = np.where(
                    type_vc_analysis['Customer Sales'] > 0,
                    (type_vc_analysis['Customer Margin $'] / type_vc_analysis['Customer Sales'] * 100),
                    0
                )
                
                type_vc_analysis = type_vc_analysis.sort_values('Customer Sales', ascending=False)
                
                # Formatage pour l'affichage
                display_type_vc = type_vc_analysis.copy()
                display_type_vc['Customer Sales'] = display_type_vc['Customer Sales'].apply(lambda x: f"{x:,.2f} €")
                display_type_vc['Customer Margin $'] = display_type_vc['Customer Margin $'].apply(lambda x: f"{x:,.2f} €")
                display_type_vc['Marge %'] = display_type_vc['Marge %'].apply(lambda x: f"{x:.2f} %")
                display_type_vc['Écart en jours'] = display_type_vc['Écart en jours'].apply(lambda x: f"{x:.1f} j" if pd.notna(x) else "N/A")
                
                # Renommer les colonnes
                display_type_vc = display_type_vc.rename(columns={
                    'Type VC': 'Type',
                    'Customer Sales': 'CA',
                    'Customer Margin $': 'Marge €',
                    'Marge %': 'Marge %',
                    'Écart en jours': 'Délai Moy.',
                    'Produit Unique': 'Nb Produits',
                    'Client Unique': 'Nb Clients'
                })
                
                st.dataframe(display_type_vc, use_container_width=True, hide_index=True)
                
                # Graphique en secteurs pour Type VC
                fig_type_vc = px.pie(
                    values=type_vc_analysis['Customer Sales'], 
                    names=type_vc_analysis['Type VC'], 
                    title="🏷️ Répartition CA par Type VC"
                )
                st.plotly_chart(fig_type_vc, use_container_width=True)
            
    with tab2:
        if len(client_df) > 0:
            # Afficher TOUS les clients et formater l'affichage
            display_df = format_dataframe_for_display(client_df, "client")
            st.dataframe(display_df, height=400, use_container_width=True, hide_index=True)
            
            # Graphiques spécifiques aux clients
            st.markdown("#### 📊 Analyses Clients")
            col1, col2 = st.columns(2)
            
            with st.spinner("📊 Génération des graphiques clients..."):
                client_figs = create_client_visualizations(filtered_df)
                if client_figs and len(client_figs) == 2:
                    fig_monthly_clients, fig_dept_clients = client_figs
                    
                    with col1:
                        st.plotly_chart(fig_monthly_clients, use_container_width=True, key="clients_monthly")
                    
                    with col2:
                        st.plotly_chart(fig_dept_clients, use_container_width=True, key="clients_dept")

    with tab3:
        if len(city_df) > 0:
            # Afficher TOUTES les villes avec toutes les informations
            display_df = format_dataframe_for_display(city_df, "city")
            st.dataframe(display_df, height=400, use_container_width=True, hide_index=True)
            # Analyse régionale
            st.markdown("#### 🗺️ Analyse par Région")
            regional_analysis = filtered_df.groupby('Région').agg({
                'Customer Sales': 'sum',
                'Customer Margin $': 'sum',
                'Client Unique': 'nunique',
                'Écart en jours': 'mean'
            }).reset_index()
            
            regional_analysis['Marge %'] = np.where(
                regional_analysis['Customer Sales'] > 0,
                (regional_analysis['Customer Margin $'] / regional_analysis['Customer Sales'] * 100),
                0
            )
            
            regional_analysis = regional_analysis.sort_values('Customer Sales', ascending=False)
            
            # Formatage pour l'affichage
            display_regional = regional_analysis.copy()
            display_regional['Customer Sales'] = display_regional['Customer Sales'].apply(lambda x: f"{x:,.2f} €")
            display_regional['Customer Margin $'] = display_regional['Customer Margin $'].apply(lambda x: f"{x:,.2f} €")
            display_regional['Marge %'] = display_regional['Marge %'].apply(lambda x: f"{x:.2f} %")
            display_regional['Écart en jours'] = display_regional['Écart en jours'].apply(lambda x: f"{x:.1f} j" if pd.notna(x) else "N/A")
            
            # Renommer les colonnes
            display_regional = display_regional.rename(columns={
                'Région': 'Région',
                'Customer Sales': 'CA',
                'Customer Margin $': 'Marge €',
                'Marge %': 'Marge %',
                'Client Unique': 'Nb Clients',
                'Écart en jours': 'Délai Moy.'
            })
            
            st.dataframe(display_regional, use_container_width=True, hide_index=True)
            
            # Graphique régional
            if len(regional_analysis) > 0:
                # AJOUTEZ CE CALCUL ICI, AVANT LA CRÉATION DES COLONNES
                ca_total_filtre = filtered_df['Customer Sales'].sum()
                regional_analysis['% CA Total'] = ((regional_analysis['Customer Sales'] / ca_total_filtre) * 100)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_regional_ca = px.bar(
                        regional_analysis, 
                        x='Région', 
                        y='Customer Sales',
                        title="💰 CA par Région",
                        color='Customer Sales',
                        color_continuous_scale='Blues'
                    )
                    fig_regional_ca.update_traces(hovertemplate='<b>%{x}</b><br>CA: %{y:,.2f}€<extra></extra>')
                    fig_regional_ca.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_regional_ca, use_container_width=True, key="cities_regional_ca")
                
                with col2:
                    fig_regional_percentage = px.bar(
                        regional_analysis, 
                        x='Région', 
                        y='% CA Total',
                        title="📊 % CA par Région",
                        color='% CA Total',
                        color_continuous_scale='Greens'
                    )
                    fig_regional_percentage.update_traces(hovertemplate='<b>%{x}</b><br>% CA Total: %{y:.2f}%<extra></extra>')
                    fig_regional_percentage.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_regional_percentage, use_container_width=True, key="cities_regional_percentage")
                
    
    # Footer avec statistiques de performance
    st.markdown("---")
    st.markdown(f"**📊 Analyse terminée** | {len(filtered_df):,} lignes analysées | {filtered_df['Produit Unique'].nunique():,} produits uniques | {filtered_df['Client Unique'].nunique():,} clients uniques")

if __name__ == "__main__":
    main()
