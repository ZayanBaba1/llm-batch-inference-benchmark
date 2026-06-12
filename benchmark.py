
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
import time

def load_model( model_name = 'gpt2' ):
  print( f'loading model { model_name }' )
  generator = pipeline( 'text-generation', model = model_name )
  print('Model Loaded')
  return generator
def run_benchmark(generator, prompt, num_sequences, num_tokens=100, num_runs=3):
    times = []
    for _ in range(num_runs):
        start = time.time()
        generator(prompt, max_new_tokens=num_tokens, num_return_sequences=num_sequences)
        end = time.time()
        times.append(end - start)
    return {
        'num_sequences': num_sequences,
        'avg_time': round(np.mean(times), 2),
        'per_sequence': round(np.mean(times) / num_sequences, 2),
        'runs': [round(t, 2) for t in times]
    }

def plot_results(results):
    sequences = [r['num_sequences'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    per_seq = [r['per_sequence'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(sequences, avg_times, color='steelblue')
    ax1.set_title('Total Inference Time by Batch Size')
    ax1.set_xlabel('Number of Sequences')
    ax1.set_ylabel('Avg Time (seconds)')
    ax1.set_xticks(sequences)

    ax2.bar(sequences, per_seq, color='coral')
    ax2.set_title('Time Per Sequence (Efficiency)')
    ax2.set_xlabel('Number of Sequences')
    ax2.set_ylabel('Avg Time per Sequence (seconds)')
    ax2.set_xticks(sequences)

    plt.suptitle('GPT-2 Inference Benchmarks (CPU, 100 tokens)', fontsize=13)
    plt.tight_layout()
    plt.savefig('benchmark.png', dpi=150)
    plt.show()
    print('Chart saved as benchmark.png!')

def main():
    generator = load_model("gpt2")
    prompt = 'The future of brain computer interface is'

    results = []
    for num_sequences in [1, 2, 3]:
        print(f'Benchmarking batch size {num_sequences}...')
        result = run_benchmark(generator, prompt, num_sequences)
        results.append(result)
        print(f"  Avg time: {result['avg_time']}s | Per sequence: {result['per_sequence']}s | Runs: {result['runs']}")

    plot_results(results)
    print('\nDone!')

if __name__ == '__main__':
    main()
