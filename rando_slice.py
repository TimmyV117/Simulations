import numpy as np
import pandas as pd

### Seeds for random numbers ###########
z = np.random.RandomState(seed=1)
simseed1 = np.random.RandomState(seed=12)
#########################################



# Random data (will be replaced with real data) ##############
# Creates arbitrary cure

random_data = {'cure': z.randint(low=1700, high=1712, size=1000),
               'X': z.normal(10, 4, 1000),
               'Y': z.normal(9, 5, 1000)}  # 11 molds
# Puts random data into dataframe and also calculates a new columns which is magnitude of X + Y vectors
rand_df = pd.DataFrame(random_data)
rand_df['MagVec'] = np.sqrt(np.square(rand_df.X) + np.square(rand_df.Y))

mold_id = rand_df.cure.unique()   #gets a llist of unique cure identities
##############################################################


sim_num_molds=[]
sim_which_molds = []
sim_magnitude_avg = []
sim_magnitude_median = []


def main():
    global sim_num_molds, sim_which_molds, sim_magnitude
    avg_mag()
    sim_num_molds.append(len(mold_id) - num_removed)
    sim_magnitude.append(avgmag)
    sim_which_molds.append(list(kept_molds))

def drop_molds():
    global mold_id, kept_molds, num_removed
    # second term in keep index is the random generation of number of molds to take out
    z.shuffle(mold_id)
    num_removed = int((simseed1.randint(0, 10) / 10) * len(mold_id))
    #kept_molds = mold_id[0:(len(mold_id)-(int((simseed1.randint(0, 10) / 10) * len(mold_id))))]
    kept_molds = mold_id[0:(len(mold_id)-num_removed)]
    return kept_molds

def avg_mag():
    global rand_df, avgmag, medianmag
    a=list(drop_molds()) #will I have to turn this into a list?
    df=rand_df[rand_df.cure.isin(a)]
    avgmag = df.MagVec.mean()
    medianmag = df.MagVec.median()
    return avgmag, medianmag





if __name__ == '__main__':
    main()

#change the number of iterations








