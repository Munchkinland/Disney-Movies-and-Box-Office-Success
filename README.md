<<<<<<< HEAD
# Disney Movies Analysis
This project started after diving deep into the dataset used on datacamp.com for this guided project https://www.datacamp.com/projects/740 Disney Movies and Box Office Success I wanted to dive deep into the original dataset. From here I created the first notebook where I made an analysis of best-selling movies and most prolific directors, but more importantly discovering all the problems related to the reliability of the original dataset. The original dataset used in the datacamp exercise is a modified version of this https://data.world/kgarrett/disney-character-success-00-16 Basically the authors used import.io to scrape the data and build the dataset.

The first consideration is about the best selling movies and can be found on this post on my blog http://www.lovabledata.com/probability-and-statistics/best-selling-disney-movies-from-1935-until-2016-python-business-analysis-with-code/
The related notebook is DisneyMoviesAndDirectorsAnalysis.ipynb 
I cleaned and merged three tables (1) Disney characters (2) box office success (3) annual gross income and plotted the results on a barplot with matplolib

Other two notebook will be deployes.
One always focused on DataViz will analyze ho changed through the time Dinsey Revenue Streams and the second one will analyze a pitfall that I think is present in the datacamp project.


Note about the Revenue Streams Data. 
From 2016 Disney Reports Changed the format of the main revenue streams.

In particular, the first change was between 2015 and 2016 in fact in 2016 they started aggregating "Consumer Products" and "Interactive". https://thewaltdisneycompany.com/app/uploads/q4-fy15-earnings.pdf
https://thewaltdisneycompany.com/walt-disney-company-reports-fourth-quarter-full-year-earnings-fiscal-2017/

Another change was made between 2018 and 2019. Here they changed the budget line "Parks and Resorts" into "Parks, Experiences and Products"
https://thewaltdisneycompany.com/the-walt-disney-company-reports-fourth-quarter-and-full-year-earnings-for-fiscal-2019/
# DisneyMoviesAnalysis

![The Walt Disney Company Revenues](https://github.com/uomodellamansarda/DisneyMoviesAnalysis/blob/main/DisneyRevenueAndreaCiufo.png)
=======
# Disney Movies Analysis

# Disney Movies and Box Office Success

Este repositorio contiene un análisis detallado del rendimiento de las películas de Disney en la taquilla, con un enfoque en diversos factores como género, clasificación, personajes, actores de voz y directores.

## Contenido

- **DisneyAnalytics.ipynb**: Cuaderno Jupyter principal que realiza el análisis.
- **data/**: Carpeta que contiene los conjuntos de datos utilizados.
- **models/**: Carpeta que almacena modelos o resultados relevantes.

## Requisitos

- Python 3
- Jupyter Notebook
- Bibliotecas de Python (puedes instalarlas mediante `pip install -r requirements.txt`)

## Cómo ejecutar el cuaderno

1. Clona este repositorio: `git clone https://github.com/Munchkinland/Disney-Movies-and-Box-Office-Success.git`
2. Ve a la carpeta del repositorio: `cd Disney-Movies-and-Box-Office-Success`
3. Instala las dependencias: `pip install -r requirements.txt`
4. Abre el cuaderno Jupyter: `jupyter notebook DisneyAnalytics.ipynb`

## Resultados

Los resultados del análisis se encuentran en [Resultados y conclusiones](#Resultados-y-conclusiones) en el cuaderno Jupyter.

## Contribuciones

Siéntete libre de contribuir abriendo problemas o enviando solicitudes de extracción.

---

**¡Disfruta explorando el mundo mágico de Disney!**

>>>>>>> e08d9f267812d9ab6fdfa288a6460fb78a2a0287
