from multiprocessing.dummy import freeze_support

import pandas as pd
import spacy
from final_files.visualisation.embeddings import spacy_embedding
from final_files.visualisation.weakest_link_method import tabulate_text, get_clusters

nlp = spacy.load("en_core_web_sm")


def identify_dates(segments):
    date_map = pd.DataFrame(index=range(0, len(segments)), columns=["Date", "Event"])

    for i, segment in enumerate(segments):
        date_map.loc[i, "Event"] = segment.strip()
        date_map.loc[i, "Date"] = ""

        doc = nlp(segment)
        # document level
        print("New segment")
        for e in doc.ents:
            if e.label_ == "DATE":
                # print(e.text, e.start_char, e.end_char, e.label_)
                curr_dates = date_map.loc[i, "Date"]
                if curr_dates == "":
                    curr_dates += e.text
                else:
                    curr_dates = curr_dates + ", " + e.text

                date_map.loc[i, "Date"] = curr_dates

        if date_map.loc[i, "Date"] == "":
            date_map.loc[i, "Date"] = "No date identified"

    date_map.to_csv("date_map.csv")
    print(date_map)
    transposed = date_map.transpose()
    print(transposed.to_json())
    return transposed.to_json()
    # ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
    # print(ents)


if __name__ == '__main__':
    freeze_support()
    dataframe = pd.read_csv('../examples/edmundsons_testing_dataset.csv')

    example = """The Bolshevik Revolution in Russia

    ‘The First World Warhad adisastrouseffecton the
    Russian economy. Agitations and anger against
    the royal family seemed to grow unchecked. By
    March 1917 the streets of Petrograd had hordes
    of people demanding peace and bread. By the
    end of March, the Czar had been abdicated
    and the Soviets had taken over. But the true
    revolution occurred when Lenin and his fellow
    Bolsheviks took over power from the unpopular
    provincial government. Lenin negotiated peace
    with Germany and left the war arena.
    
    
    Depiction of Bolshevik Revolution by an artist
    
       
    
    The American Entry into the War
    
    When the War broke out in Europe, the USA had decided to remain neutral as it felt it was
    a European war which had nothing to do with it. The American President, Woodrow Wilson
    had asked Americans to be ‘impartial in thought as well as in action’. But Germany in an
    attempt to break the British power decided to use its navy to isolate Britain. The German
    submarines began to attack all merchant ships that crossed the Atlantic.
    
    In early 1917, the German U-boats sank the Lusitania which had a number of Americans on
    board. USA broke-off diplomatic relations with Germany, and when it intercepted a message from Germany urging Mexico to join the war and reclaim land that it had lost to America,
    it decided to declare war on Germany. These events changed the outcome of the war. Fresh
    enthusiastic American troops helped boost the morale of the Allies and push the war in
    their favour.
    
    THE EFFECTS AND CONSEQUENCES OF THE WAR
    
    Military Casualties
    
    The ‘Great War’, which began on July 28, 1914 and ended with the German armistice of
    November 11, 1918, had resulted in a vast number of casualties and deaths and similarly vast
    numbers of missing soldiers. The precise numbers remain shrouded in the passage of time
    compounded by the incompleteness of available records."""

    clusters = get_clusters(example.replace("\n", "").replace("  ", " "), 3, spacy_embedding)

    print(clusters)
    identify_dates(clusters)
    # Example 1
    # t1_org = dataframe['original'][0]
    # t2_org = dataframe['original'][1]
    #
    # clus_real1 = dataframe['paragraphs'][0].strip().split("\n")
    # clus_real2 = dataframe['paragraphs'][1].strip().split("\n")
    #
    #
    # clus_pred1 = get_clusters(t1_org, 3, spacy)

    # print(table.transpose().to_json())
    # print(evaluate(clus_real1, clus_pred1))
    # print(evaluate(clus_real2, clus_pred2))
    #
    # table.to_csv("table-1-bert.csv")
    # # Example 2
    # t1_org = dataframe['original'][4]
    # t2_org = dataframe['original'][5]
    #
    # clus_real1 = dataframe['paragraphs'][4].strip().split("\n")
    # clus_real2 = dataframe['paragraphs'][5].strip().split("\n")
    #
    # clus_pred1, clus_pred2, table = tabulate_text(t1_org, t2_org, 2, doc2vec)
    #
    # print(table.transpose().to_json())
    # print(evaluate(clus_real1, clus_pred1))
    # print(evaluate(clus_real2, clus_pred2))
    # table.to_csv("table-2-bert.csv")
