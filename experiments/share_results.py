# The goal of this function is to share the user result and add them inside registered benchmark results
import os
import pandas as pd
import tkinter as tk


def share_results(dataset_id):
    # try:
    #     app = Application(dataset_id)
    # except:
        print('\n [QUESTION] Do you want to share/merge your sampler results for this dataset inside the benchmark ?')
        res = input('Yes = 1/ No = 0 :')
        
        while res not in ['1', '0']:
            print("[INPUT ERROR] Please chose an answer between '0' and '1'")

            print('\n [QUESTION] Do you want to share/merge your sampler results for this dataset inside the benchmark ?')
            res = input('Yes = 1/ No = 0 :')

        if res=='1':
            merge_results(dataset_id=dataset_id)
            print('[INFO] Results saved')
        else:
            print('[INFO] Results not saved')



def merge_results(dataset_id):
    # Add all user csv results inside registered csv results
    dir_path = f'user_results/results_{dataset_id}/db/'

    list_csv_filenames = [ f for f in os.listdir(dir_path) if ( 
        os.path.isfile(os.path.join(dir_path,f)) and  # It's a file
        '.csv' in os.path.splitext(f)[1] and        # File extension is csv
        'indexes' not in os.path.splitext(f)[0]     # File is not the indexes.csv file
        ) ]

    # for dataset_id in dataset_ids:
    for filename in list_csv_filenames:
        user_df = pd.read_csv(f'user_results/results_{dataset_id}/db/{filename}')
        registered_df = pd.read_csv(f'Experiments/results_{dataset_id}/db/{filename}')

        df = pd.concat([registered_df, user_df], ignore_index=True)
        df.to_csv(f'Experiments/results_{dataset_id}/db/{filename}', index=False)



class Application(tk.Tk):
    """ Ask user if he wants to share his results """
    def __init__(self, dataset_id):
        tk.Tk.__init__(self)

        print("AAAA")
        self.label = tk.Label(self, text="Do you want to share/merge your sampler results for this dataset inside the benchmark")
        self.yes_button = tk.Button(self, text="Yes", fg="Green", command=self.merge_results)
        self.no_button = tk.Button(self, text="No", fg="Red", command=self.destroy)
        self.label.pack()
        self.yes_button.pack()
        self.no_button.pack()

        self.dataset_id = dataset_id

    def merge_results(self):
        merge_results(self.dataset_id)
        self.destroy


def undo_merge(dataset_id, method_name):

    dir_path = f'results_{dataset_id}/db/'

    list_csv_filenames = [ f for f in os.listdir(dir_path) if ( 
        os.path.isfile(os.path.join(dir_path,f)) and  # It's a file
        '.csv' in os.path.splitext(f)[1] and        # File extension is csv
        'indexes' not in os.path.splitext(f)[0]     # File is not the indexes.csv file
        ) ]

    # for dataset_id in dataset_ids:
    for filename in list_csv_filenames:
        df = pd.read_csv(f'results_{dataset_id}/db/{filename}')
        # df = pd.concat([registered_df, user_df], ignore_index=True)
        df = df[df['method'] != method_name]
        df.to_csv(f'results_{dataset_id}/db/{filename}', index=False)



if __name__ == '__main__':
    # Run this from benchmark_tabular_active_learning folder
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id', help='Dataset to process')
    parser.add_argument('method_name', help='Method results to remove from benchmark results')
    args = parser.parse_args()
    dataset_id = args.dataset_id
    method_name = args.method_name

    undo_merge(dataset_id, method_name)
