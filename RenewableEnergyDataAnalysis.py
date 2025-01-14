import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # for file operations
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def collect_data():
    """
    Collect data from multiple sources and combine them into single source
    """
    countries = [
        "USA",
        "China",
        "India",
        "Brazil",
        "Germany",
        "France",
        "Italy",
        "Spain",
        "UK",
        "Australia",
    ]

    data = {
        "country": countries,
        "solar_capacity_gw": np.random.uniform(10, 100, 10),
        "wind_capacity_gw": np.random.uniform(20, 150, 10),
        "hydro_capacity_gw": np.random.uniform(30, 200, 10),
        "carbon_emission_mt": np.random.uniform(100, 1000, 10),
        "gdp_billions": np.random.uniform(500, 20000, 10),
        "population_millions": np.random.uniform(20, 1400, 10),
    }

    return pd.DataFrame(data)


def clean_data(df):
    """
    Clean and prepare data for data analysis
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Create derived features
    df["total_renewable_capacity"] = (
        df["solar_capacity_gw"] + df["wind_capacity_gw"] + df["hydro_capacity_gw"]
    )

    df["emission_per_capita"] = (df["carbon_emission_mt"] / df["population_millions"])

    df["renewable_per_gdp"] = (df["total_renewable_capacity"] / df["gdp_billions"])

    return df

def perform_analysis(df):
    """
       Perform stastical analysis and generate insights
    """

    #Calculate correlation matrix
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_columns].corr()

    #Peform clustering analysis
    features_for_clustering = ["renewable_per_gdp", "emission_per_capita"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features_for_clustering])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled_features)

    #Calculate summary stats
    summary_stats = df[numeric_columns].describe()

    return corr_matrix, summary_stats, df

def visualize_results(df, corr_matrix):
    """
    Visualize the results of the analysis
    """

    #set up the plotting style
    plt.style.use('ggplot')


    #Create Correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True, cmap='coolwarm',center=0)
    plt.title("Correlation Matrix of Key Metrics")

    #Create scatter plot of emissions vs renewable capacity
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df,x='total_renewable_capacity', y="emission_per_capita", hue='cluster')
    plt.title("Emissions per capita vs Total Renewable Capacity")

    #Create bar plot of renewable capacity by country
    plt.figure(figsize=(12,6))
    df.plot(kind="bar", x='country', y=['solar_capacity_gw', 'wind_capacity_gw', 'hydro_capacity_gw'], stacked=True)
    plt.title("Renewable Capacity by Country")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def generate_insights(df):
    """
        Generate key insights and recommendations based on the analysis
    """    
    insights = []

    #Analyze renewable adoption patterns
    top_renewable = df.nlargest(3, 'total_renewable_capacity')
    insights.append(f"Top 3 Countries by renewable capacity: {','.join(top_renewable['country'].tolist())}")

    #Analyze emission patterns
    emission_correlation = df['emission_per_capita'].corr(df['renewable_per_gdp'])
    insights.append(f"Correlation between emissions and renewable adoption:{emission_correlation:.2f}")

    #Analyze the economic relationships
    gdp_correlation = df['gdp_billions'].corr(df['total_renewable_capacity'])
    insights.append(f"Correlation between GDP and renewable adoption:{gdp_correlation:.2f}")

    return insights

def main():
    #Execute the entire data analysis pipeline
    raw_data = collect_data()
    clean_df = clean_data(raw_data)
    corr_matrix, summary_stats, analyzed_df = perform_analysis(clean_df)
    visualize_results(analyzed_df, corr_matrix)
    insights = generate_insights(analyzed_df)

    return insights, analyzed_df

if __name__ == "__main__":
     insights, final_df = main()
     print("\n Key Instights:")
     for insight in insights:
        print(f"- {insight}")
