import sqlite3
from random import randint, uniform

import model

# create database
conn = sqlite3.connect('exps.db')
c = conn.cursor()
c.execute('create table if not exists experiments (exp_id integer, layers integer, layersizes text, lr real)')
c.execute('create table if not exists results (exp_id integer, epoch integer, val_loss real, val_acc real)')
conn.commit()

# generate random hyperparameters
num_exps = 1000
layer_cfgs = [[randint(32, 128) for i in range(randint(1, 3))] for j in range(num_exps)]
learning_rates = [uniform(0.003, 0.045) for i in range(num_exps)]

# experiment loop
for layers, lr in zip(layer_cfgs, learning_rates):
    c.execute('select max(exp_id) from experiments')

    # increment experiment id, then add new row to experiments table
    exp_id = c.fetchone()[0]
    exp_id = 1 + exp_id if exp_id is not None else 0
    c.execute('insert into experiments values (?, ?, ?, ?)', (exp_id, len(layers), ' '.join(map(str, layers)), lr))
    conn.commit()

    print('running experiment {}: {}, {}'.format(exp_id, layers, lr))
    model.run_exp(layers, lr, conn, exp_id)

conn.close()

