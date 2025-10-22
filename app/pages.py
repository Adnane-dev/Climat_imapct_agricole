import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def show_overview(data):
    """Affiche la vue d'ensemble des données climatiques"""
    st.header("📊 Vue d'ensemble des données climatiques")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Période couverte",
            f"{data['full']['YEAR'].min()} - {data['full']['YEAR'].max()}"
        )
    
    with col2:
        st.metric(
            "Stations météo",
            f"{data['station_avg']['ID'].nunique():,}"
        )
    
    with col3:
        st.metric(
            "Régions",
            f"{data['annual_regional']['REGION'].nunique()}"
        )
    
    with col4:
        st.metric(
            "Points de données",
            f"{len(data['full']):,}"
        )

def show_pandas_analysis(data):
    """Affiche les analyses basées sur Pandas"""
    st.header("📈 Analyse des données avec Pandas")
    
    # Tendances annuelles
    st.subheader("Évolution des températures moyennes")
    fig = px.line(
        data['annual_trends'],
        x='YEAR',
        y=['TMIN', 'TAVG', 'TMAX'],
        title="Évolution des températures (2000-2025)"
    )
    st.plotly_chart(fig)
    
    # Distribution des précipitations
    st.subheader("Distribution des précipitations par région")
    fig = px.box(
        data['annual_regional'],
        x='REGION',
        y='PRCP',
        title="Distribution des précipitations par région"
    )
    st.plotly_chart(fig)

def show_matplotlib_viz(data):
    """Affiche les visualisations spécialisées avec Matplotlib"""
    st.header("🎨 Visualisations spécialisées avec Matplotlib")
    
    # Créer une figure Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique 1: Evolution des températures
    data['annual_trends'].plot(
        x='YEAR',
        y=['TMIN', 'TAVG', 'TMAX'],
        ax=ax1,
        title="Évolution des températures"
    )
    
    # Graphique 2: Précipitations annuelles
    data['annual_trends'].plot(
        x='YEAR',
        y='PRCP',
        kind='bar',
        ax=ax2,
        title="Précipitations annuelles"
    )
    
    plt.tight_layout()
    st.pyplot(fig)

def show_seaborn_analysis(data):
    """Affiche les visualisations statistiques avec Seaborn"""
    st.header("📉 Visualisations statistiques avec Seaborn")
    
    # Configuration du style Seaborn
    sns.set_style("whitegrid")
    
    # Créer une figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribution des températures
    sns.boxplot(
        data=data['annual_regional'],
        x='REGION',
        y='TAVG',
        ax=ax1
    )
    ax1.set_title("Distribution des températures par région")
    
    # Corrélation Température/Précipitations
    sns.scatterplot(
        data=data['annual_regional'],
        x='TAVG',
        y='PRCP',
        hue='REGION',
        ax=ax2
    )
    ax2.set_title("Relation Température/Précipitations")
    
    plt.tight_layout()
    st.pyplot(fig)

def show_plotly_interactive(data):
    """Affiche les visualisations interactives avec Plotly"""
    st.header("🔄 Visualisations interactives avec Plotly")
    
    # Carte des stations
    fig = px.scatter_mapbox(
        data['station_avg'],
        lat='LATITUDE',
        lon='LONGITUDE',
        color='TAVG',
        size='PRCP',
        hover_name='NAME',
        zoom=5,
        title="Carte des stations météorologiques"
    )
    fig.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig)
    
    # Graphique temporel interactif
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['monthly']['YEAR'],
        y=data['monthly']['TAVG'],
        mode='lines+markers',
        name='Température moyenne'
    ))
    fig.update_layout(title="Évolution temporelle interactive")
    st.plotly_chart(fig)

def show_regional_analysis(data):
    """Affiche l'analyse par région"""
    st.header("🗺️ Analyse régionale")
    
    # Comparaison des régions
    st.subheader("Comparaison des indicateurs par région")
    
    # Température moyenne par région
    fig = px.bar(
        data['annual_regional'],
        x='REGION',
        y='TAVG',
        color='REGION',
        title="Température moyenne par région"
    )
    st.plotly_chart(fig)
    
    # Précipitations par région
    fig = px.bar(
        data['annual_regional'],
        x='REGION',
        y='PRCP',
        color='REGION',
        title="Précipitations par région"
    )
    st.plotly_chart(fig)

def show_agricultural_impact(data):
    """Affiche l'analyse de l'impact agricole"""
    st.header("🌾 Impact sur l'agriculture")
    
    # Indicateurs agricoles
    st.subheader("Indicateurs agricoles clés")
    
    # Growing Degree Days
    fig = px.line(
        data['annual_trends'],
        x='YEAR',
        y='GDD',
        title="Growing Degree Days (GDD)"
    )
    st.plotly_chart(fig)
    
    # Risques climatiques
    fig = px.bar(
        data['annual_trends'],
        x='YEAR',
        y=['FROST_RISK', 'HEAT_WAVE', 'DRY_DAY'],
        title="Risques climatiques"
    )
    st.plotly_chart(fig)