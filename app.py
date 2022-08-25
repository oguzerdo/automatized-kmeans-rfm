import pandas as pd
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.loader import upload_gsheet
from scripts.data_prep import *

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# LOAD DATA
print("Loading Data")
df = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
print("Data Loaded!")


def get_namespace():
    parser = argparse.ArgumentParser()

    # manual olarak çalıştırmak istersek
    # arguman verilmezse otomatik clustering yapılır.
    # ön tanımlı değeri false.
    # python ..py --manual
    parser.add_argument('--manual', dest='manual', action='store_true')
    parser.set_defaults(manual=False)

    parser.add_argument('--upload', dest='upload', action='store_true')
    parser.set_defaults(manual=False)

    return parser.parse_args()


def data_prep(dataframe):
    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    return dataframe


def create_rfm(dataframe, uniqid, recency, freq, monetary):
    # Date optimization
    dataframe[recency] = pd.to_datetime(dataframe[recency]).dt.date
    last_day = dataframe[recency].max().day
    today_date = dataframe[recency].max().replace(day=last_day + 2)

    # RFM METRIKLERININ HESAPLANMASI

    rfm = dataframe.groupby(uniqid).agg({recency: lambda date: (today_date - date.max()).days,
                                         freq: lambda num: num.nunique(),
                                         monetary: lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    return rfm


def kmeans_clustering(dataframe, clustering="auto", graph=True):
    scaler = MinMaxScaler()
    segment_data = pd.DataFrame(scaler.fit_transform(dataframe[["recency", "frequency", "monetary"]]),
                                index=dataframe.index)
    segment_data.columns = ["Recency_n", "Frequency_n", "Monetary_n"]

    kmeans = KMeans()
    ssd = []
    K = range(1, 30)

    if clustering == "manual":
        print("Manual Mode")
        for k in K:
            kmeans = KMeans(n_clusters=k).fit(segment_data)
            ssd.append(kmeans.inertia_)

        ssd

        plt.plot(K, ssd, "bx-")
        plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
        plt.title("Optimum Küme sayısı için Elbow Yöntemi")
        plt.savefig("outputs/elbow-graph.png", dpi=250)
        plt.show(block=False)
        input_message = print("Enter cluster number:")
        try:
            cluster_number = int(input(input_message))
        except:
            print("Hatalı Giriş!")
        kmeans = KMeans(n_clusters=cluster_number).fit(segment_data)

        segment_data["clusters"] = kmeans.labels_

    else:
        print("Automatic Mode")
        kmeans = KMeans()
        elbow = KElbowVisualizer(kmeans, k=(2, 20))
        elbow.fit(segment_data)
        elbow.show()
        kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(segment_data)
        segment_data["clusters"] = kmeans.labels_
        print(f"Number of cluster selected: {elbow.elbow_value_}")

    if graph:
        x = segment_data[['Recency_n', 'Frequency_n', 'Monetary_n']].values
        model = KMeans(n_clusters=kmeans.n_clusters, random_state=0)
        y_clusters = model.fit_predict(x)

        import random
        number_of_colors = model.n_clusters

        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0, model.n_clusters):
            ax.scatter(x[y_clusters == i, 0], x[y_clusters == i, 1], x[y_clusters == i, 2], s=40, color=color[i],
                       label=f"cluster {i}")

        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        fig.savefig(f"outputs/{model.n_clusters}-cluster.png", dpi=250)
        plt.show(block=False)
    return segment_data


def merge(rfm, segment_data):
    segmentation = rfm.merge(segment_data, on="Customer ID")
    seg_desc = segmentation[["segment", "clusters", "recency", "frequency", "monetary"]].groupby(
        ["clusters", "segment"]).agg(["mean", "count"])
    print(seg_desc)
    return segmentation


def main(manual, upload):
    rfm_data = data_prep(df)
    print("RFM DATA İŞLENDİ")
    rfm = create_rfm(rfm_data, uniqid="Customer ID", recency="InvoiceDate", freq="Invoice", monetary="TotalPrice")
    print("RFM HAZIRLANDI")

    if manual:
        segment_data = kmeans_clustering(rfm, clustering="manual")
    else:
        segment_data = kmeans_clustering(rfm, clustering="auto")

    segmentation = merge(rfm, segment_data)
    segmentation.reset_index(drop=False, inplace=True)

    segmentation['Recency_n'] = segmentation['Recency_n'].apply(lambda x: round(x, 2))
    segmentation['Frequency_n']= segmentation['Frequency_n'].apply(lambda x: round(x, 2))
    segmentation['Monetary_n'] = segmentation['Monetary_n'].apply(lambda x: round(x, 2))

    if upload:
        print('Upload Mode')
        upload_gsheet(segmentation)
    else:
        pass

    return segmentation


if __name__ == "__main__":
    namespace = get_namespace()
    segmentation = main(manual=namespace.manual, upload=namespace.upload)
    segmentation.to_csv("outputs/segmentation.csv")



