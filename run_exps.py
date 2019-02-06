'''
run_exps.py: a simple experiment script that establishes a connection with a
SQLite3 database, then performs a random hyperparameter search.

Elliott Skomski (skomsks@wwu.edu)
'''

import sqlite3
from random import randint, uniform

import model

conn = sqlite3.connect('exps.db') # create DB or load if it already exists

# get cursor to make queries, then add necessary tables if they don't exist
cur = conn.cursor()
cur.execute('create table if not exists experiments (exp_id integer, layers integer, \
           layersizes text, lr real)')
cur.execute('create table if not exists results (exp_id integer, epoch integer, \
           train_loss real, train_acc real, val_loss real, val_acc real)')
cur.connection.commit()

# get largest experiment ID from DB
cur.execute('select max(exp_id) from experiments')
exp_id = cur.fetchone()[0]
exp_id = 0 if exp_id is None else exp_id + 1

# experiment loop
num_exps = 1000
for _ in range(num_exps):
    # generate random hyperparameters
    layers = [randint(32, 128) for i in range(randint(1, 3))]
    lr = uniform(0.003, 0.045)

    # insert new experiment info into DB
    cur.execute('insert into experiments values (?, ?, ?, ?)',
              (exp_id, len(layers), ' '.join(map(str, layers)), lr))
    cur.connection.commit()

    # run the experiment
    print('running experiment {}: {}, {}'.format(exp_id, layers, lr))
    model.run_exp(layers, lr, cur, exp_id)

    exp_id += 1

conn.close()

