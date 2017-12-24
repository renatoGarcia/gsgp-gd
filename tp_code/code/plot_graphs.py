#!/usr/bin/env python

import pandas as pd
import click
import seaborn as sns
from pathlib import Path
from plot_utils import plot_setup, figure_size, time_plot
import matplotlib.pyplot as plt
import re
import joblib


mem = joblib.Memory(cachedir='/tmp', verbose=1)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


@click.group()
def cli():
    plot_setup()


@mem.cache
def read_files(files_path):
    r = re.compile('/run_(?P<config>\w+)/(?P<dataset>.*)-sp0\.(?P<p_start>\d)alp(?P<alpha>\d+)')
    path = Path(files_path)

    dframes = []
    for sdir in path.iterdir():
        for dataset in sdir.iterdir():
            if not dataset.is_dir() or dataset.name == 'scripts':
                continue

            m = r.search(str(dataset))
            df1 = (pd.read_csv(dataset/'tsFitness.csv', header=None)
                   .melt(id_vars=[0], var_name='i_iter', value_name='test_fitness')
                   .rename(columns={0: 'i_run'}))
            df2 = (pd.read_csv(dataset/'trFitness.csv', header=None)
                   .melt(id_vars=[0], var_name='i_iter', value_name='train_fitness')
                   .rename(columns={0: 'i_run'}))
            df = pd.merge(df1, df2, on=['i_iter', 'i_run'])

            df.loc[:, 'config'] = m['config']
            df.loc[:, 'dataset'] = m['dataset']
            df.loc[:, 'p_start'] = m['p_start']
            df.loc[:, 'alpha'] = m['alpha']
            dframes.append(df)

    df = pd.concat(dframes)
    return df


@cli.command()
@click.argument('inputs', type=click.Path(resolve_path=True), nargs=-1)
@click.option('--width-scale', default=1.0)
@click.option('--text-width', default=390.0)
@click.option('--output', type=click.Path(resolve_path=True))
def funca_timeserie(inputs, width_scale, text_width, output):
    width, height = figure_size(width_scale, text_width=text_width)
    # height *= 1.2

    df = read_files(*inputs)
    df = (df.query('dataset=="func_a"')
          .filter(['i_run', 'i_iter', 'test_fitness', 'config'])
          .rename(columns={'test_fitness': 'fitness', 'i_iter': 'iteração'})
          .replace({'funca_pall': 'pall', 'funca_aall': 'aall'}))

    dfa = (df.query('config=="aall"'))
    dfb = (df.query('config=="pall"'))

    plt.figure(figsize=(width, height))
    time_plot(dfa.iteração, dfa.fitness, dfa.i_run, label='analítico')
    time_plot(dfb.iteração, dfb.fitness, dfb.i_run, label='protegido')
    plt.xlim(0, 2000)
    plt.legend()

    if output:
        plt.savefig(output, bbox_inches='tight')
    else:
        plt.show()


@cli.command()
@click.argument('inputs', type=click.Path(resolve_path=True), nargs=-1)
@click.option('--width-scale', default=1.0)
@click.option('--text-width', default=390.0)
@click.option('--output', type=click.Path(resolve_path=True))
def funca_confidence_width(inputs, width_scale, text_width, output):
    width, height = figure_size(width_scale, text_width=text_width)

    df = read_files(*inputs)
    df = (df.query('dataset=="func_a" and i_iter>0')
          .filter(['i_run', 'i_iter', 'test_fitness', 'config'])
          .replace({'funca_pall': 'pall', 'funca_aall': 'aall'}))

    def bla(x):
        import statsmodels.stats.api as sms
        aa = sms.DescrStatsW(x)
        a, b = aa.tconfint_mean(alpha=0.05)
        return b-a

    a = (df.filter(['i_iter', 'test_fitness', 'config'])
         .groupby(['config', 'i_iter'])
         .aggregate(bla)
         .reset_index())

    df = a.rename(columns={'test_fitness': 'fitness', 'i_iter': 'iteração'})

    dfa = (df.query('config=="aall"'))
    dfb = (df.query('config=="pall"'))

    plt.figure(figsize=(width, height))
    plt.plot(dfa.iteração, dfa.fitness,  ls='-',
             label='analítico')
    plt.plot(dfb.iteração, dfb.fitness,  ls='-',
             label='protegido')
    plt.xlabel('Iteração')
    plt.ylabel('Largura da confiança')
    plt.xlim(0, 2001)
    plt.legend()

    if output:
        plt.savefig(output, bbox_inches='tight')
    else:
        plt.show()


@cli.command()
@click.argument('inputs', type=click.Path(resolve_path=True), nargs=-1)
@click.option('--width-scale', default=1.0)
@click.option('--text-width', default=390.0)
@click.option('--output', type=click.Path(resolve_path=True))
def confidence_width(inputs, width_scale, text_width, output):
    width, height = figure_size(width_scale, text_width=text_width)

    df = read_files(*inputs)
    df = (df.query('dataset!="func_a" and i_iter==2001')
          .filter(['test_fitness', 'config', 'dataset']))

    def bla(x):
        import statsmodels.stats.api as sms
        aa = sms.DescrStatsW(x)
        a, b = aa.tconfint_mean(alpha=0.05)
        return b-a

    df = (df.filter(['test_fitness', 'config', 'dataset'])
          .groupby(['dataset', 'config'])
          .aggregate(bla)
          .reset_index()
          .rename(columns={'test_fitness': 'Largura da confiança'}))
    df['Operadores'] = df.loc[:, 'config'].replace({'pall': 'protegido',
                                                    'pdiv': 'protegido',
                                                    'aall': 'analítico',
                                                    'adiv': 'analítico'})
    df['Funções'] = df.loc[:, 'config'].replace({'pall': 'todas',
                                                 'pdiv': 'divisão',
                                                 'aall': 'todas',
                                                 'adiv': 'divisão'})

    sns.factorplot(x='Operadores', y='Largura da confiança', kind='point',
                   hue='Funções', order=['protegido', 'analítico'],
                   legend=True,
                   col='dataset', col_wrap=4, data=df,
                   sharey=False, size=height, aspect=1)

    if output:
        plt.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('inputs', type=click.Path(resolve_path=True), nargs=-1)
@click.option('--width-scale', default=1.0)
@click.option('--text-width', default=390.0)
@click.option('--output', type=click.Path(resolve_path=True))
def final_state(inputs, width_scale, text_width, output):
    width, height = figure_size(width_scale, text_width=text_width)

    df = read_files(*inputs)
    df = (df.query('i_iter==2001 and dataset != "func_a"')
          .rename(columns={'test_fitness': 'fitness'}))

    sns.factorplot(x='config', y='fitness', kind='bar',
                   col='dataset', col_wrap=4, data=df,
                   sharey=False, size=height, aspect=1)

    if output:
        plt.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('inputs', type=click.Path(resolve_path=True), nargs=-1)
@click.option('--width-scale', default=1.0)
@click.option('--text-width', default=390.0)
@click.option('--output', nargs=2, type=click.Path(resolve_path=True))
def timeseries(inputs, width_scale, text_width, output):
    # width, height = figure_size(width_scale, height_scale=0.1, text_width=text_width)
    width, height = figure_size(width_scale, text_width=text_width)
    height *= 0.6

    df = read_files(*inputs)
    df = (df.query('dataset!="func_a"')
          .rename(columns={'test_fitness': 'fitness', 'i_iter': 'iteração'}))

    a = df.dataset.unique()
    for idx, b in enumerate(chunker(a, 8)):
        dfa = df.query('@b in dataset')

        hue_order = df.config.unique()
        (sns.FacetGrid(dfa, col='dataset', hue='config', hue_order=hue_order,
                       col_wrap=2, sharey=False, size=height, aspect=1.5)
         .map(time_plot, 'iteração', 'fitness', 'i_run')
         .add_legend())

        if output:
            plt.savefig(output[idx])
        else:
            plt.show()


if __name__ == '__main__':
    cli()
