import argparse
import pandas as pd
import os
from common import add_filepath_suffix, newspaper_from_path


def validate_batch_indices(sample, full_extractions):
    '''Require a one-to-one match before joining or deleting checkpoints.'''
    if not sample.index.is_unique:
        raise ValueError("Template index contains duplicate row ids.")
    if not full_extractions.index.is_unique:
        raise ValueError(
            "Extraction batches contain duplicate row ids (overlapping batches).")
    if len(full_extractions) != len(sample):
        raise ValueError(
            "Found {} extraction rows for {} template rows: batches are missing "
            "or --batch_size/--skip do not match the producing run.".format(
                len(full_extractions), len(sample)))

    missing = sample.index.difference(full_extractions.index)
    extra = full_extractions.index.difference(sample.index)
    if len(missing) or len(extra):
        raise ValueError(
            "Batch row ids do not match the template (missing {}, extra {})."
            .format(missing[:5].to_list(), extra[:5].to_list()))

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
    batch_frames, batch_files = [], []
    nbatches = args.nbatches or (len(sample) // args.batch_size + 1)
    print("Iterating over {} batches.".format(nbatches))
    for batch_idx in range(nbatches):
        batch = args.batch_size * (batch_idx + 1) + args.skip
        file = os.path.join(args.batch_dir, '-'.join([newspaper, args.suffix, 'batch', str(batch)]) + '.gzip')
        if not os.path.isfile(file):
            print("File not found: '{}'".format(file))
            break
        batch_frames.append(pd.read_parquet(file))
        batch_files.append(file)
        print("After batch {}, have {} extraction rows.".format(
            file, sum(len(f) for f in batch_frames)))

    full_extractions = pd.concat(batch_frames) if batch_frames else pd.DataFrame()
    validate_batch_indices(sample, full_extractions)
    sample = sample.join(full_extractions)
    if not sample.index.is_unique or len(sample) != len(full_extractions):
        raise RuntimeError(
            "Join did not preserve the validated one-row-per-id template.")

    # Write full data to file
    # sample.to_csv(add_filepath_suffix(args.output_dir, newspaper, n=len(sample), suffix='extract-wage', ext='csv'))        
    print("Have final shape of {}.".format(sample.shape))
    output_file = add_filepath_suffix(args.output_dir, newspaper, n=len(sample),
        suffix='{}-merged'.format(args.suffix))
    sample.to_parquet(output_file, compression='gzip')
    if not os.path.isfile(output_file):
        raise RuntimeError("Merged output was not written; keeping batches.")

    if args.delete:
        # Delete exactly the checkpoints that were read and validated, and only
        # after the merged parquet has been written successfully.
        for file in batch_files:
            os.remove(file)
            print("Removed batch {}.".format(file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True,
        help="Filepath to template data.")
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

    if not os.path.isfile(args.filepath):
        parser.error('Invalid filepath to template data.')
    if not os.path.isdir(args.batch_dir):
        parser.error('Invalid filepath to batch directory.')
    # The output directory is ours to create; asserting on it only
    # made a first run fail on a path the user never chose.
    os.makedirs(args.output_dir, exist_ok=True)
    main()
