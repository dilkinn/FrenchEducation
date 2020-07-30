import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
from scipy import stats

"""
Block of functions helps to clean and join dataframes.
"""

def keep_rows_without_remarks(data, col='Remarque'):
    '''
    Removes 'bad' rows (those that has something else rather that NaN in col) from
    pandas dataframe (data).

    Input:
    -----
    data : pd.DataFrame
    col  : str
            column name

    Returns:
    _______
    modified_data : pd.DataFrame
    '''
    return data.loc[data[col].isna()]


def keep_columns(data, cols):
    '''
    Returns dataframe with only columns from cols list.

    Input:
    -----
    data : pd.DataFrame
    cols  : list of strings
            columns names

    Returns:
    _______
    modified_data : pd.DataFrame
    '''
    return data.iloc[:, cols]

def load_data():
    '''
    Loads datasets using pd.read_scv function. Treats delimeter and na values.

    Input:
    -----

    Returns:
    -------
    Pandas data frames.
    '''
    df_doc = pd.read_csv('data_univ/insertion-professionnelle-des-diplomes-de-doctorat-par-ensemble.csv',
                         delimiter=';', na_values=('ns','nd','.'))
    df_mas = pd.read_csv('data_univ/fr-esr-insertion_professionnelle-master.csv',
                         delimiter=';', na_values=('ns','nd','.'))
    df_lp = pd.read_csv('data_univ/fr-esr-insertion_professionnelle-lp.csv',
                        delimiter=';', na_values=('ns','nd','.'))
    univ_names = pd.read_csv('data_univ/fr-esr-principaux-etablissements-enseignement-superieur.csv',
                             delimiter=";", na_values=('ns','nd','.'))
    return df_doc, df_mas, df_lp, univ_names

def rename_cols(name_dct, df):
    '''
    Renames columns of dataframe df using dictionary name_dct.

    Input:
    -----
    name_dct : dictionary
                Names in string format to be changed.
    df : pd.DataFrame
                Dataframe to perform changes on.

    Returns:
    -------
    df : pd.DaraFrame
        Data frame with columns renamed.
    '''
    df.rename(columns=name_dct, errors="raise", inplace=True)
    return df

def concat_dfs(dfs_lst):
    '''
    Merges data frames from the list dfs_lst by rows.

    Input:
    -----
    dfs_lst : list of pd.DateFrames
                List of data frames to concatenate by rows.

    Returns:
    -------
    df : pd.DaraFrame
        Merged data frame.
    '''
    df = dfs_lst[0]
    for i in range(1, len(dfs_lst)):
        df = pd.concat([df, dfs_lst[i]])
    return df

def df_clean(df, cols_to_keep):
    '''
    Reduces dataframe df by keeping columns from list columns col_to_keep,
    removing lines with any kind of remarks in "Remarque" column.
    Two new columns created: "Diploma Domain" and
    "taux temp plein - out of number of answers".

    Input:
    -----
    cols_to_keep : list of str
                Names of columns to keep in df.
    df : pd.DataFrame
                Dataframe to perform changes on.

    Returns:
    -------
    df : pd.DaraFrame
        Data frame with columns in cols_to_keep plus two more.
    '''
    df_new = df
    df_new = keep_rows_without_remarks(df_new)
    #df_new = df_new[df_new["Taux d’insertion"]].dropna()
    df_new = keep_columns(df_new, cols_to_keep)
    df_new["Diploma Domain"] = df_new["Diplôme"] + " " + df_new["Domaine"]
    df_new["taux temp plein - out of number of answers"] = df_new["Taux d’insertion"] * df_new["% emplois à temps plein"]/100
    return df_new

def merge_uni_all(univ_df, main_df):
    '''
    Left join of dataframes: main_df and univ_df. Adding information about universities based
    on "uai - identifiant" column.

    Input:
    -----
    univ_df : pd.DataFrame
                Data frame containing universities info.
    main_df : pd.DataFrame
                Dataframe to join with.

    Returns:
    -------
    out_df : pd.DaraFrame
        Joined data frame.
    '''
    out_df = main_df.copy()
    univ_red = univ_df[["uai - identifiant","Libellé",
                           "type d'établissement","Secteur d'établissement",
                           "Géolocalisation"]]

    out_df = pd.merge(out_df, univ_red, how='left',
                      left_on="Numéro de l'établissement",
                      right_on="uai - identifiant",
                      left_index=False, right_index=False)
    return out_df

def doc_name_dct():
    '''
    This function returns a dictionary to properly rename some columns.
    This helps to stay consistent within dataframes.
    '''
    dct = {"NUMERO_UAI_Courant": "Numéro de l'établissement",
           "Année d’obtention": "Annee",
           "Situation": "situation",
           "Part de femmes diplômées":"% femmes",
           "Nombre de répondant":"Nombre de réponses",
           "Part de docteurs déclarant avoir été financés pour réaliser leur thèse":"% de diplômés boursiers",
           "Taux d’emploi stable":"% emplois stables",
           "Taux d’emploi cadre":"% emplois cadre",
           "Taux d’emploi à temps plein":"% emplois à temps plein",
           "Salaire net mensuel médian des emplois à temps plein":"Salaire net médian des emplois à temps plein",
           "Salaire brut annuel moyen estimé":"Salaire brut annuel estimé",
           "Taux d'insertion":"Taux d’insertion",
           "Discipline principale agrégée de l’école doctorale":"Domaine"
          }
    return dct

"""
Block of functions that helps plotting.
"""


def tperiod(n):
    """
    Return dictionary with column names describing the time period.
    """
    dct = {1: ['12 mois après le diplôme', '18 mois après le diplôme'],
           3: ['30 mois après le diplôme', '36 mois après le diplôme']}
    return dct[n]


def col_name_by_key(key):
    """
    Returns column names given key.

    Input:
    -----
    key : str
        Key value to access column name.

    Returns:
    -------
    value : str
        Related column name.
    """
    dct = {'job placement': "Taux d’insertion",
           'salary': "Salaire net médian des emplois à temps plein",
           'full time': "% emplois à temps plein"}
    return dct[key]


def d_domains():
    '''
    Return list of strings with degree and domain information.
    Corresponds to all possible individual values of "Diploma domain" column.
    '''
    d_doms = [
        'DOCTORAT Sciences du vivant',
        'DOCTORAT Sciences humaines et humanités',
        'DOCTORAT Sciences et leurs interactions',
        'MASTER ENS Masters enseignement',
        'MASTER LMD Droit, économie et gestion',
        'MASTER LMD Lettres, langues, arts',
        'MASTER LMD Sciences humaines et sociales',
        'MASTER LMD Sciences, technologies et santé',
        'LICENCE PRO Droit, économie et gestion',
        'LICENCE PRO Lettres, langues, arts',
        'LICENCE PRO Sciences humaines et sociales',
        'LICENCE PRO Sciences, technologies et santé']
    return d_doms


def lbls(key1, key2):
    """
    Return labels for plotting.

    Input:
    -----
    key1 : str
            First key.
    key2 : str
            Second key.

    Return:
    ------
    value : str
            Label.
    """
    #     plot_labels = {1: {"job placement": {'lims': (60, 101, 5),
    #                       'label': "Job placement rates, %",
    #                       'title': 'Job placement: 12-18 months after graduation'},
    #                 "salary": {'lims': (1000, 2900, 250),
    #                   'label': "Net monthly salary in euros",
    #                   'title': 'Full time job monthly salaries: 12-18 months after graduation'}},
    #               3: {"job placement": {'lims': (60, 101, 5),
    #                    'label': "Job placement rates, %",
    #                    'title': 'Job placement: 30-36 months after graduation'},
    #                   "salary": {'lims': (1000, 2900, 250),
    #                              'label': "Net monthly salary in euros",
    #                              'title': 'Full time job monthly salaries: 30-36 months after graduation'}}}
    plot_labels = {"job placement": {'lims': (60, 101, 5),
                                     'label': "Job placement rates, %",
                                     'title': 'Job placement: 3 years after graduation'},
                   "salary": {'lims': (1000, 2900, 250),
                              'label': "Net monthly salary in euros",
                              'title': 'Full time job monthly salaries: 3 years after graduation'}}
    return plot_labels[key1][key2]


def domain_labels():
    """
    Returns list of labels.
    """
    labels = [['Life Sciences',
               'Social and Humanity Sciences',
               'Sciences and Their Interactions'],
              ['Masters Teaching',
               'Law, Economics, Management',
               'Lettres, Langues, Arts',
               'Social and Humanity Sciences',
               'Science, Technology and Health'],
              ['Law, Economics, Management',
               'Letters, Languages, Arts',
               'Social and Humanity Sciences',
               'Science, Technology and Health']]
    return labels


def domain_colors():
    """
    Returns colors.
    """
    colors = [['lightgreen', 'lightyellow', 'lightblue'],
              ['orchid', 'darkorange', 'salmon', 'lightyellow', 'lightblue'],
              ['orange', 'salmon', 'lightyellow', 'lightblue']]
    return colors


def cond_sit(df, time_per, key):
    """
    Return condition to mask data frame and choose only those
    that correspond certain time period.

    Input:
    -----
    df : pandas.DataFrame
            Data frame to create a mask from.
    time_per : int
            1 or 3 choose (12 and 18) or (30 and 36) months after graduation, respectively.
    key : int
            0 or 1 to choose (12 or 18)/(30 or 36).

    Return:
    ------
    value : pd.Series of boolean
            Mask choosing necessary values.
    """
    return df["situation"] == tperiod(time_per)[key]


def cond_dd(df, d_domain):
    """
    Returns a mask choosing d_domain.

    Input:
    -----
    df : pd.DataFrame
    d_domain : str
            String to mask

    Returns:
    -------
    value: pd.Series of boolean
            Mask choosing necessary rows.
    """
    return df["Diploma Domain"] == d_domain


def dfs_to_plot(df, plot_par, time_per):
    """
    Returns data for plotting.

    Input:
    -----
    df : pd.DataFrame
            Data frame with data.
    plot_par : str
            Plot parameter key. ("salary" or "job placement")
    time_per : int
            Time period choice key: 1 or 3 (years after graduation).

    Return:
    ------
    output : list of pd.DataFrames.
            Data to plot.
    """
    cond_1 = cond_sit(df, time_per, 0)
    cond_2 = cond_sit(df, time_per, 1)
    cond_3 = lambda x: cond_dd(df, x)
    cond = cond_1 | cond_2

    s_col = col_name_by_key(plot_par)

    return [df[(cond_3(d_domain) & cond)][s_col].dropna() for d_domain in d_domains()]


def sub_titles():
    """
    Returns labels for sub plots.
    """
    return ["DOC", "MAS", 'BAC']


def ax_plot(df_plots, ax, i, f_s, x_lbls, y_lims, plot_par, time_per, alpha):
    """
    Populates axes ax[i] using data and parameters.
    """
    if alpha < 1:
        box = ax[i].boxplot(df_plots[i], 0, ".", 0,
                            labels=domain_labels()[i],
                            patch_artist=True,
                            boxprops=dict(alpha=alpha),
                            whiskerprops=dict(alpha=alpha))
    else:
        box = ax[i].boxplot(df_plots[i], 0, "+", 0,
                            labels=domain_labels()[i],
                            patch_artist=True,
                            boxprops=dict(alpha=alpha),
                            whiskerprops=dict(alpha=alpha))

    for patch, color in zip(box['boxes'], domain_colors()[i]):
        patch.set_facecolor(color)
    ax[i].tick_params(axis='both', which='minor', labelsize=f_s)
    ax[i].set_title(x_lbls[i], fontsize=f_s)
    ax[i].set_xlim([y_lims[0], y_lims[1]])
    ax[i].set_yticklabels(domain_labels()[i], fontsize=f_s)  # rotation=90,
    if i == 0:
        ax[0].set_xticklabels([])
    elif i == 1:
        ax[1].set_xticklabels([])
    elif i == 2:
        ax[2].set_xlabel(lbls(plot_par, "label"), fontsize=f_s)
        ax[2].set_xticklabels(list(range(y_lims[0], y_lims[1], y_lims[2])), fontsize=f_s)

def plot_domain_wise(df, plot_par, filename):
    """
    Plots figure using data df and parameter plot_par. Saves figure to filename.

    Input:
    -----
    df : pd.DataFrame
        Data frame with data
    plot_par : str
        key to select column name: "salary" or "job placement"
    filename : str
        filename to save
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))

    for time_per, alpha in zip([1, 3], [0.3, 1]):
        df_sals = dfs_to_plot(df, plot_par, time_per)
        df_plots = [df_sals[:3], df_sals[3:8], df_sals[8:]]

        f_s = 12
        x_lbls = sub_titles()
        y_lims = lbls(plot_par, 'lims')

        for i in range(len(domain_labels())):
            ax_plot(df_plots, ax, i, f_s, x_lbls, y_lims, plot_par, time_per, alpha)

    fig.suptitle(lbls(plot_par, "title"), fontsize=f_s + 2)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()

# Ranking of universities
def univ_ranking(df):
    """
    Adds a column "rank" to the dataframe. And populates it with ranks of
    universities from top 20.

    Input:
    -----
    df : pd.DataFrame
        data
    """
    ranks_dct = {
        'Université Sorbonne Paris Nord':2,
        'Aix-Marseille Université': 4,
        'Université Grenoble Alpes':5,
        'Université de Strasbourg':6,
        'Université de Montpellier':8,
        'Université Claude Bernard - Lyon 1':9,
        'Université de Toulouse 3 - Paul Sabatier':10,
        'Université de Bordeaux':11,
        'Université Toulouse 1 - Capitole':12,
        'Université de Lille':18,
        'Université de Lorraine':19
    }
    df['rank'] = np.nan
    for uni in ranks_dct.keys():
        df.loc[df["Libellé"] == uni,"rank"] = ranks_dct[uni]

# Folium and maps
def univ_locations(df, filename="univ_locations.html"):
    '''
    Locates universities on the map of France.

    Input:
    -----
    df : pd.DataFrame
        data
    filename : str
        output filemane *.html

    Return:
    ------
    m : folium map image
    '''
    m = folium.Map(location=[47.0000, 2.0000], zoom_start=5.0, width=400, height=400)
    for point in np.unique(df['Géolocalisation'].dropna()):
        geo_point = tuple(map(float, point.split(',')))
        folium.Marker(tuple(geo_point)).add_to(m)
    m.save(filename)
    return m


def students_num(df, time_per=1):
    '''
    Creates a subset of data in order to visualize the number of students.

    Input:
    -----
    df : pd.DataFrame
    time_per : int
            1 or 3 for (12 amd 18) or (30 or 36) months after graduation.

    Return:
    ------
    df_map_ : pd.DataFrame
            grouped and aggregated data ready for visualization.
    '''
    cond_1 = cond_sit(df, time_per, 0)
    cond_2 = cond_sit(df, time_per, 1)
    cond = cond_1 | cond_2

    cols_to_keep = [22, 19, 8, 16, 0, 10, 12]

    df_map = keep_columns(df, cols_to_keep)
    df_map = df_map.loc[cond].dropna()

    cols_gr = ["Géolocalisation", "Libellé", "Diploma Domain", "Annee"]

    df_map_ = pd.DataFrame(df_map.groupby(by=cols_gr)['Nombre de réponses'].sum())
    df_map_['mean salary'] = df_map.groupby(by=cols_gr)[col_name_by_key("salary")] \
        .mean()
    df_map_["avg job placement rate"] = df_map.groupby(by=cols_gr) \
        [col_name_by_key("job placement")].mean()
    df_map_.reset_index(inplace=True)
    return df_map_

def master_domains():
    """
    Returns a list of diploma domain pairs for Masters degree.
    """
    # This specific order provides best visibility.
    # It follows the descending order of number of students.
    m_d = ['MASTER LMD Droit, économie et gestion',
          'MASTER LMD Sciences, technologies et santé',
          'MASTER LMD Sciences humaines et sociales',
          'MASTER ENS Masters enseignement',
          'MASTER LMD Lettres, langues, arts']
    return m_d

def master_colors():
    """
    Returns colors representing master_domains() list of diploma domains.
    """
    m_c = ['darkorange', 'lightblue', 'lightyellow', 'orchid', 'salmon']
    return m_c

def students_distr_fig(df, year=2016, filename="france_nst_distr.html"):
    """
    Add dots on the map. Size of those dots represents number of students
    at each university locations. Color of dots represents different domains.

    Input:
    -----
    df : pd.DataFrame
            Data frame with data
    year : int
            year to represent
    filename : str
            output filename *.html

    Output:
    ------
    m : folium map image
    """
    m = folium.Map(location=[47.0000, 2.0000], zoom_start=5.0, width=400, height=400)
    for j, d_domain in enumerate(master_domains()):
        df_map = df[(df["Diploma Domain"] == d_domain) & (df["Annee"] == year)]
        for i in range(0, len(df_map)):
            point = df_map.iloc[i]["Géolocalisation"]
            geo_point = tuple(map(float, point.split(',')))
            folium.Circle(
                location=geo_point,
                popup=df_map.iloc[i]['Libellé'],
                radius=df_map.iloc[i]['Nombre de réponses']*50,
                color=master_colors()[j],
                fill=True,
                fill_color=master_colors()[j]
            ).add_to(m)

    m.save(filename)
    return m



def students_jbplcmt_fig(df, year=2016, filename="job_placement_distr.html"):
    """
    Add dots on the map. Size of those dots represents job placement rate
    at each university locations. Color of dots represents different domains.

    Input:
    -----
    df : pd.DataFrame
            Data frame with data
    year : int
            year to represent
    filename : str
            output filename *.html

    Output:
    ------
    m : folium map image
    """
    m = folium.Map(location=[47.0000, 2.0000], zoom_start=5.0, width=400, height=400)
    for j, d_domain in enumerate(master_domains()):
        df_map = df[(df["Diploma Domain"] == d_domain) & (df["Annee"] == year)]
        for i in range(0, len(df_map)):
            point = df_map.iloc[i]["Géolocalisation"]
            geo_point = tuple(map(float, point.split(',')))
            folium.Circle(
                location=geo_point,
                popup=df_map.iloc[i]['Libellé'],
                radius=df_map.iloc[i]['avg job placement rate']*300,
                color=master_colors()[j],
                fill=True,
                fill_color=master_colors()[j]
            ).add_to(m)

    m.save(filename)
    return m

def paris_area_ntot(df, year=2016, filename='paris_st_distr.html'):
    location = geolocator.geocode("Paris")
    print(location.latitude, location.longitude)
    m = folium.Map(location=(location.latitude, location.longitude), zoom_start=14.0, width=600, height=400)

    for j, d_domain in enumerate(master_domains()):
        df_map = df[(df["Diploma Domain"] == d_domain) & (df["Annee"] == year)]

        for i in range(0,len(df_map)):
            point = df_map.iloc[i]["Géolocalisation"]
            geo_point = tuple(map(float, point.split(',')))
            folium.Circle(
                location=geo_point,
                popup=df_map.iloc[i]['Libellé'],
                radius=df_map.iloc[i]['Nombre de réponses']*0.3,
                color=master_colors()[j],
                fill=True,
                fill_color=master_colors()[j]
            ).add_to(m)

    # Save it as html
    m.save(filename)
    return m

def calc_n_st(df):
    '''
    Create three more columns which represent number of students with jobs, with full time
    jobs and total rate of full time job holders.
    '''
    df["students number w_job"] = df["Nombre de réponses"] \
                                    * df["Taux d’insertion"] / 100
    df["students number w_fulltime_job"] = df["students number w_job"] \
                                    * df["% emplois à temps plein"] / 100
    df["taux temp plein - out of number of answers"] = df["Taux d’insertion"] \
                                    * df["% emplois à temps plein"] / 100

def p_val(df1, df2, col_name="weighted job placement"):
    data_1 = df1[col_name].dropna()
    data_2 = df2[col_name].dropna()
    stat, p = stats.ttest_ind(data_1, data_2)
    if p > 0.05:
        print("We fail to reject the null hypothesis. Our datasets are not proven to be statistically different")
    elif p < 0.05:
        print("We reject the null hypothesis. Our datasets are statistically different")
    print(f'Means for column {col_name}:')
    print(f'1: {data_1.mean()}')
    print(f'2: {data_2.mean()}')
    print(f'p-value: {p}')
    return p


def weighted_dfs(dds, df, time_per=3):
    h1 = []
    for i, dd in enumerate(dds):
        cond_1 = cond_sit(df, time_per, 0)
        cond_2 = cond_sit(df, time_per, 1)

        df_h = df[(df["Diploma Domain"] == dd) \
                  & (cond_1 | cond_2)][:]  # .dropna()

        nsum = df_h["Nombre de réponses"].sum()

        df_h["weighted job placement"] = df_h["students number w_job"] / nsum
        df_h["weighted ft job placement"] = df_h["students number w_fulltime_job"] / nsum

        h1.append(df_h)
    return h1


def ranked_dfs(dds, df, time_per=3):
    cond_1 = cond_sit(df, time_per, 0)
    cond_2 = cond_sit(df, time_per, 1)
    conds = [df["rank"].notnull(), df["rank"].isnull()]

    h2 = []
    for dd in dds:
        ranked = []
        for cond in conds:
            df_r = df[(df["Diploma Domain"] == dd) \
                      & (cond_1 | cond_2) & (cond)][:]

            nsum = df_r["Nombre de réponses"].sum()

            df_r["weighted job placement"] = df_r["students number w_job"] / nsum
            df_r["weighted ft job placement"] = df_r["students number w_fulltime_job"] / nsum
            ranked.append(df_r)
        h2.append(ranked)

    return h2

if __name__ == '__main__':
    # Block 1 : Cleaning and joining dataframes.
    df_doc, df_mas, df_lp, univ_names = load_data()
    df_doc = rename_cols(doc_name_dct(), df_doc)
    df_all = concat_dfs([df_lp, df_mas, df_doc])
    # Keeping only informative for this project columns
    cols_to_keep = [0, 1, 2, 5, 7, 9, 10, 11, 12, 13, 15, 18, 19, 20, 23, 26]
    df_all = df_clean(df_all, cols_to_keep)
    df_all = merge_uni_all(univ_names, df_all)
    print(f'Total number of answers from students: {df_all["Nombre de réponses"].dropna().sum()}')
    print(f'Numver of universities: {df_all["Libellé"].value_counts().count()}')

    # Block 2: Plotting.
    plot_domain_wise(df_all, "job placement", "images/job_placement.png")
    plot_domain_wise(df_all, "salary", "images/salary.png")

    # Universities ranking
    univ_ranking(df_all)

    # Folium
    univ_locations(df_all, filename="images/univ_locations.html") # ranks of universities
    df_st_num = students_num(df_all, time_per=1) # number of students per location per year
    students_distr_fig(df_st_num, filename="images/students_distr.html")
    students_jbplcmt_fig(df_st_num, filename="images/job_placement_distr.html")

    # Hypothesis testing
    calc_n_st(df_all)
    cols_to_keep = [0, 1, 4, 6, 8, 9, 10, 11, 16, 17, 23, 24, 25]
    df_hyp1 = keep_columns(df_all, cols_to_keep)
    cols_to_keep = [0, 1, 4, 6, 8, 9, 10, 11, 16, 22, 23, 24, 25]
    df_hyp2 = f.keep_columns(df_all, cols_to_keep)
    dds = [f.d_domains()[-1], f.d_domains()[7]]

    # Hypothesis 1 : Higher level of education give better chances to get (permanent) job
    h1 = weighted_dfs(dds, df_hyp1, time_per=3)
    # Any job placement
    p_val(h1[0], h1[1], col_name="weighted job placement")
    p_val(h1[0], h1[1], col_name="Taux d’insertion")
    # Full-time job placement
    p_val(h1[0], h1[1], col_name="weighted ft job placement")
    p_val(h1[0], h1[1], col_name="taux temp plein - out of number of answers")

    # Hypothesis 2 : Higher ranked schools give same job placement as the rest of schools
    h2 = f.ranked_dfs(dds, df_hyp2, time_per=3)
    ranks = ['h', 'l']
    for i, dd in enumerate(dds):
        print(f'Diploma domain:{dd}')
        p_val(h2[i][0], h2[i][1], col_name="weighted job placement")
        p_val(h2[i][0], h2[i][1], col_name="weighted ft job placement")


