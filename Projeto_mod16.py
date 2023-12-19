import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from itertools import product
from sklearn import tree
import io
 
# Configure a página do Streamlit
st.set_page_config(page_title="Análise Exploratória de Dados", page_icon="✨", layout="wide")
st.title("Previsão de Renda com a Metodologia CRISP-DM: Visualização e Análise 📊💰")

# Função para carregar dados e remover colunas desnecessárias
def load_data(file_path, cols_to_exclude):
    # Feedback para o usuário durante o carregamento dos dados
    st.text("Carregando dados... Por favor, aguarde.")
    try:
        return pd.read_csv(file_path, usecols=lambda col: col not in cols_to_exclude).drop_duplicates()
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")

# Função para exibir informações sobre remoção de duplicatas e o DataFrame
def display_duplicates_and_dataframe(renda):
    duplicatas = renda.duplicated().sum()
    st.write(f'Foram removidas {duplicatas} duplicatas.' if duplicatas > 0 else 'Não há duplicatas no DataFrame.')
    st.write(f'Quantidade após remoção das linhas duplicadas: {len(renda)}\n')
    st.subheader("Visualização do DataFrame:")
    st.dataframe(renda)

# Função para exibir o relatório de perfil
def display_profiling_report(renda):
    st.title("YData Profiling - Análise Exploratória de Dados")
    prof = ProfileReport(renda, minimal=False, explorative=True, dark_mode=True, orange_mode=True, correlations={"pearson": {"calculate": False}})
    st.components.v1.html(prof.to_html(), height=15000)

# Função para exibir estatísticas descritivas e correlação para colunas numéricas
def display_descriptive_and_correlation(renda):
    numeric_renda = renda.select_dtypes(include='number')
    correlation_result = numeric_renda.describe().T.join(numeric_renda.corr().iloc[-1:])
    st.subheader("Estatísticas Descritivas e Correlação:")
    st.write(correlation_result)

# Função para plotar contagem categórica
def plot_categorical_count(renda):
    if not renda.empty:
        coluna_de_interesse = st.selectbox("Selecione a coluna para o gráfico de contagem:", renda.columns)
        st.subheader("Gráfico de Contagem por Categoria:")
        st.bar_chart(renda[coluna_de_interesse].value_counts())

# Carregar dados e remover colunas desnecessárias
file_path = r"C:\Users\Arthu\OneDrive\Área de Trabalho\Data Science\notebooks\previsao_de_renda.csv"
cols_to_exclude = ['Unnamed: 0', 'id_cliente']
renda = load_data(file_path, cols_to_exclude)

# Exibir informações sobre remoção de duplicatas e o DataFrame
display_duplicates_and_dataframe(renda)

# Gráfico de contagem para colunas categóricas
plot_categorical_count(renda)

# Exibir o relatório de perfil
display_profiling_report(renda)

# Estatísticas descritivas e correlação de variáveis quantitativas
display_descriptive_and_correlation(renda)

# Pairplot
st.subheader("Pairplot:")
if renda is not None:
    pairplot = sns.pairplot(data=renda, hue='tipo_renda', vars=['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda'], diag_kind='hist', height=2)
    st.pyplot(pairplot)
else:
    st.warning("DataFrame 'renda' é None. Por favor, verifique seu processo de carregamento de dados.")

# Função para plotar clustermap
def plot_clustermap(renda):
    st.subheader("Clustermap para Colunas Numéricas:")
    clustermap_data = renda.select_dtypes(include='number').corr()
    cluster_grid = sns.clustermap(data=clustermap_data, center=0, cmap=sns.diverging_palette(h_neg=100, h_pos=359, as_cmap=True, sep=1, center='light'))
    fig_clustermap = cluster_grid.fig
    ax_clustermap = cluster_grid.ax_heatmap
    st.pyplot(fig_clustermap)

# Chamar a função quando necessário
plot_clustermap(renda)

# Função para plotar scatter plot com linha de regressão
def plot_scatter_and_regression(renda):
    st.subheader("Scatter Plot com Linha de Regressão:")
    scatter_plot, ax_scatter = plt.subplots(figsize=(16, 9))
    sns.scatterplot(x='tempo_emprego', y='renda', hue='tipo_renda', size='idade', data=renda, alpha=0.4, ax=ax_scatter)
    sns.regplot(x='tempo_emprego', y='renda', data=renda, scatter=False, color='.3', ax=ax_scatter)
    st.pyplot(scatter_plot.figure)

# Função para plotar point plots
def plot_point_plots(renda):
    st.subheader("Point Plots:")
    sns.set(rc={'figure.figsize': (12, 4)})

    # Criar subplots para point plots
    fig, axes = plt.subplots(1, 2)

    # Point plot para 'posse_de_imovel'
    sns.pointplot(x='posse_de_imovel', y='renda', data=renda, dodge=True, ax=axes[0])

    # Point plot para 'posse_de_veiculo'
    sns.pointplot(x='posse_de_veiculo', y='renda', data=renda, dodge=True, ax=axes[1])

    # Exibir os gráficos usando Streamlit
    st.pyplot(fig)
     
# Função para plotar colunas qualitativas
def plot_qualitative_columns(renda, qualitativas):
    for col in qualitativas:
        # Criar subplots
        fig, axes = plt.subplots(1, 2, gridspec_kw=common_params)

        # Stacked bar plot
        renda_crosstab = pd.crosstab(index=renda['data_ref'], columns=renda[col], normalize='index')
        ax0 = renda_crosstab.plot.bar(stacked=True, ax=axes[0])
        ax0.set_xticklabels(labels=renda['data_ref'].dt.strftime('%b/%Y').unique(), rotation=45)
        ax0.legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")

        # Point plot para renda média ao longo do tempo
        sns.pointplot(x='data_ref', y='renda', hue=col, data=renda, dodge=True, errorbar=('ci', 95), ax=axes[1])
        axes[1].set_xticks(range(len(renda['data_ref'].unique())))  # Ajustado

# Converter 'data_ref' para datetime
renda['data_ref'] = pd.to_datetime(renda['data_ref'])
qualitativas = renda.select_dtypes(include=['object', 'boolean']).columns

# Configurar parâmetros comuns de estilo
common_params = {'width_ratios': [1, 1.5], 'wspace': 0.6}

# Configurar tamanho da figura para todos os plots
plt.rc('figure', figsize=(16, 4))

# Loop através de colunas qualitativas para subplots
for col in qualitativas:
    # Criar subplots
    fig, axes = plt.subplots(1, 2, gridspec_kw=common_params)

    # Stacked bar plot
    renda_crosstab = pd.crosstab(index=renda['data_ref'], columns=renda[col], normalize='index')
    ax0 = renda_crosstab.plot.bar(stacked=True, ax=axes[0])
    ax0.set_xticklabels(labels=renda['data_ref'].dt.strftime('%b/%Y').unique(), rotation=45)
    ax0.legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")

    # Point plot para renda média ao longo do tempo
    sns.pointplot(x='data_ref', y='renda', hue=col, data=renda, dodge=True, errorbar=('ci', 95), ax=axes[1])
    axes[1].set_xticks(range(len(renda['data_ref'].unique())))  # Ajustado
    axes[1].set_xticklabels(labels=renda['data_ref'].dt.strftime('%b/%Y').unique(), rotation=45)
    axes[1].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")

    # Exibir os gráficos usando Streamlit
    st.pyplot(fig)

# Descartar a coluna 'data_ref' se presente
renda.drop(columns='data_ref', inplace=True, errors='ignore')

# Descartar linhas com valores ausentes
renda.dropna(inplace=True)

# Dados resumidos
st.subheader("Dados Resumidos:")
summary_data = {
    'Tipo de Dado': renda.dtypes,
    'Valores Não Nulos': renda.count(),
    'Categorias Únicas': renda.nunique()
}
st.table(summary_data)

# Análise de Correlação
renda_dummies = pd.get_dummies(renda)
correlation_result = renda_dummies.corr()['renda'].sort_values(ascending=False).to_frame(name='Correlação').reset_index()

# Exibir Análise de Correlação
st.subheader("Análise de Correlação:")
st.dataframe(correlation_result.style.bar(color=['darkred', 'darkgreen'], align='zero'))

# Matriz de características (X) e variável alvo (y)
X = renda_dummies.drop(columns='renda')
y = renda_dummies['renda']

# Exibir o formato de X e y
st.write(f'Formato de X: {X.shape}, comprimento de y: {len(y)}')

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Exibir os formatos dos conjuntos de treinamento e teste
st.write(f'Conjunto de treinamento: Formato de X_train: {X_train.shape}, formato de y_train: {y_train.shape}')
st.write(f'Conjunto de teste: Formato de X_test: {X_test.shape}, formato de y_test: {y_test.shape}')

# Criar uma lista de combinações de hiperparâmetros
hyperparameters = list(product(range(1, 21), range(1, 31)))

# Criar um DataFrame com hiperparâmetros
score = pd.DataFrame({
    'max_depth': [max_depth for max_depth, min_samples_leaf in hyperparameters],
    'min_samples_leaf': [min_samples_leaf for max_depth, min_samples_leaf in hyperparameters],
    'score': [DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
              .fit(X_train, y_train)
              .score(X_test, y_test) for max_depth, min_samples_leaf in hyperparameters]
})

# Ordenar o DataFrame por 'score' em ordem decrescente
score = score.sort_values(by='score', ascending=False)

# Exibir o DataFrame ordenado
st.write(score)

# Treinar o modelo de árvore de decisão com hiperparâmetros ótimos
reg_tree = DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_leaf=4)
reg_tree.fit(X_train, y_train)

# Visualização da Árvore de Decisão para X_train e X combinados
st.subheader("Visualização da Árvore de Decisão:")
fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(reg_tree, feature_names=list(X_train.columns) + list(X.columns), filled=True, rounded=True, fontsize=8, ax=ax)
st.pyplot(fig)

# Representação de texto da árvore de decisão
st.subheader("Representação de Texto da Árvore de Decisão:")
st.text(tree.export_text(reg_tree))

# Avaliar o modelo nos conjuntos de treinamento e teste
r2_train = reg_tree.score(X_train, y_train)
r2_test = reg_tree.score(X_test, y_test)

# Exibir os resultados usando f-strings
template = f'O coeficiente de determinação (𝑅2) da árvore com profundidade = {reg_tree.get_depth()} para a base de {{}} é: {{:.2f}}'

st.write(template.format('treino', r2_train).replace(".", ","))
st.write(template.format('teste', r2_test).replace(".", ","), '\n')

# Adicionar os valores previstos ao DataFrame
renda['renda_predict'] = np.round(reg_tree.predict(X), 2)

# Exibir os valores reais e previstos
st.subheader("Valores Reais e Previstos:")
st.write(renda[['renda', 'renda_predict']])

# Dados de entrada para previsão
st.subheader("Dados de Entrada para Previsão:")
entrada_data = {
    'sexo': 'M',
    'posse_de_veiculo': False,
    'posse_de_imovel': True,
    'qtd_filhos': 1,
    'tipo_renda': 'Assalariado',
    'educacao': 'Superior completo',
    'estado_civil': 'Solteiro',
    'tipo_residencia': 'Casa',
    'idade': 34,
    'tempo_emprego': None,
    'qt_pessoas_residencia': 1
}

# Criar um DataFrame a partir dos dados de entrada
entrada = pd.DataFrame([entrada_data])

# Pré-processar os dados de entrada e fazer a previsão
entrada = pd.get_dummies(entrada).reindex(columns=X.columns, fill_value=0)
entrada = pd.concat([X, entrada], sort=False).fillna(0).tail(1)

# Fazer a previsão
predicted_renda = np.round(reg_tree.predict(entrada), 2)

# Se houver várias previsões, pegue a primeira
if isinstance(predicted_renda, np.ndarray):
    predicted_renda = predicted_renda[0]

# Exibir a renda estimada
st.subheader("Renda Estimada:")
st.write(f"Renda estimada: R${predicted_renda:.2f}".replace('.', ','))