# communications_neural_net
Implementation of Neural Nets for Communications Channel Decoding using Log Likelihood Ratios

Implementation and experimentation of Machine LLRning - Learning to Softly Demodulate by Ori Shental and Jacob Hoydis (Bell Labs)

The primary purpose of this project was to test and compare the results of demodulating signals using traditional demodulation methods using exact/approximate log-likelihood ratios, and the proposed neural net demodulator. The secondary purpose was to write the program such that the simulated communication network and the LLRnet can be easily customized and tested with a relatively simple interface.

Following is the Pipeline for the Transmitter:

Random Serial Bit Generator -> Serial to Parallel -> Apply Modulation Scheme -> Separate into Imaginary and Real Components + Add Noise

Following is the Pipeline for the Receiver and Decoder:

Compute Log Likelihood Ratios -> Convert Log Likelihood Ratio to bit -> Convert Parallel to Serial

