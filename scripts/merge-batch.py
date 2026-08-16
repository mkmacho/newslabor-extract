import argparse
import pandas as pd
import os
from common import add_filepath_suffix, newspaper_from_path

def main():
    ''' Concatenate and join batched extractions. '''

    # Load data
    nrows = args.batch_size * args.nbatches if args.nbatches else None

    if args.filepath.endswith('.gzip'):
        sample = pd.read_parquet(args.filepath, columns=args.cols)
        if nrows: sample = sample.iloc[:nrows]
    else:
        # index_col is applied *after* usecols, so the unnamed index column must
        # be requested explicitly; otherwise the first requested column silently
        # becomes the index and the join below yields all-NaN columns.
        usecols = args.cols
        if usecols:
            header = pd.read_csv(args.filepath, nrows=0)
            usecols = [header.columns[0]] + [c for c in usecols if c != header.columns[0]]
        sample = pd.read_csv(args.filepath, nrows=nrows, usecols=usecols, index_col=[0])
    if 'raw_content' in sample.columns:
        sample.raw_content = sample.raw_content.fillna('')

    if args.skip:
        sample = sample.iloc[args.skip:]
    print("Loaded template data of {} rows.".format(len(sample)))

    newspaper = newspaper_from_path(args.filepath)

    # Concatenate extraction batches
    batch_frames = []
    nbatches = args.nbatches or (len(sample) // args.batch_size + 1)
    print("Iterating over {} batches.".format(nbatches))
    for batch_idx in range(nbatches):
        batch = args.batch_size * (batch_idx + 1) + args.skip
        file = os.path.join(args.batch_dir, '-'.join([newspaper, args.suffix, 'batch', str(batch)]) + '.gzip')
        if not os.path.isfile(file):
            print("File not found: '{}'".format(file))
            break
        batch_frames.append(pd.read_parquet(file))
        print("After batch {}, have {} extraction rows.".format(
            file, sum(len(f) for f in batch_frames)))

    full_extractions = pd.concat(batch_frames) if batch_frames else pd.DataFrame()
    assert len(full_extractions) == len(sample), (
        "Found {} extraction rows for {} template rows: batches are missing or "
        "--batch_size/--skip do not match the producing run.".format(
            len(full_extractions), len(sample)))
    sample = sample.join(full_extractions)

    # Write full data to file
    # sample.to_csv(add_filepath_suffix(args.output_dir, newspaper, n=len(sample), suffix='extract-wage', ext='csv'))        
    print("Have final shape of {}.".format(sample.shape))
    sample.to_parquet(add_filepath_suffix(args.output_dir, newspaper, n=len(sample), 
        suffix='{}-merged'.format(args.suffix)), compression='gzip')

    if args.delete:
        for batch_idx in range(nbatches):
            batch = args.batch_size * (batch_idx + 1) + args.skip
            file = os.path.join(args.batch_dir, '-'.join([newspaper, args.suffix, 'batch', str(batch)]) + '.gzip')
            if not os.path.isfile(file): 
                break
            os.remove(file)
            print("Removed batch {}.".format(file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="Filepath to template data.")
    parser.add_argument('--batch_dir', type=str, help="Filepath to directory of batches",
        default='./output')
    parser.add_argument('-n', '--nbatches', type=int, default=None, help="Limit size.")
    parser.add_argument('-s', '--suffix', type=str, default='resolve', help="Batches of what.")
    parser.add_argument('-b', '--batch_size', type=int, default=10000, help="Batch size.")
    parser.add_argument('--cols', action='append', default=None, help="Columns to read from batch.")
    # Default 0: deleting the checkpoints is destructive (they can represent days
    # of paid API calls) and must be an explicit choice.
    parser.add_argument('-d', '--delete', type=int, default=0, help="Delete batches.")
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('-o', '--output_dir', type=str, help="Filepath to output directory.",
        default='./output')
    
    args = parser.parse_args()

    assert os.path.isfile(args.filepath), 'Invalid filepath to template data.'
    assert os.path.isdir(args.batch_dir), 'Invalid filepath to batch directory.'
    # The output directory is ours to create; asserting on it only
    # made a first run fail on a path the user never chose.
    os.makedirs(args.output_dir, exist_ok=True)
    main()

