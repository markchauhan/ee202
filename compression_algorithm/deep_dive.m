%part 1
filename = 'deep_dive.wav';
[audioSignal, fs] = audioread(filename);
%sound(audioSignal, fs);

downloadFile = matlab.internal.examples.downloadSupportFile("audio","wav2vec2/wav2vec2-base-960.zip");
wav2vecLocation = fullfile(tempdir, "wav2vec");
unzip(downloadFile, wav2vecLocation);
addpath(wav2vecLocation);

%part 2
clientObj = speechClient("wav2vec2.0");
disp(clientObj);

%part 3
transcript = speech2text(clientObj, audioSignal,fs);
disp(transcript);

%part 4
fft_audio = fft(audioSignal);
length_audio = length(audioSignal);
%display(fft_audio);

%part 5 + 6 + 7 + 8

%loop go through decreasing cutoff values
%zero out the FFT beyond frequencey
%inverse FFT
start_cutoff = 10000; %20kHz
end_cutoff = 1000; % 1kHz
step_cutoff = -100; %decrease by 1kHz

%arrays to store results
frequencies = [];
averages = [];
failed_transcript = '';
failed_confidence = '';
failed_cutoff = 0;
%og signal
sound(audioSignal, fs); % Play og signal

for cutoff = start_cutoff:step_cutoff:end_cutoff
        cutoff_con = floor(cutoff / (fs / length_audio));

        %zero out frequencies
        compressed_fft = fft_audio;
        compressed_fft(cutoff_con+1:end-cutoff_con+1) = 0;
    
        % Preserve conjugate symmetry for real signal
        compressed_fft(length_audio-cutoff_con+2:length_audio) = conj(compressed_fft(cutoff_con:-1:2));

        %taking inverses
        compressed_audio = ifft(compressed_fft, 'symmetric');

        %normalizing
        compressed_audio = compressed_audio / max(abs(compressed_audio));

        %apply model to get new transcription
        compressed_transcript = speech2text(clientObj, compressed_audio, fs);
        disp(['Compressed Transcript with Cutoff ' num2str(cutoff) ' Hz:']);
        disp(compressed_transcript);

        frequencies = [frequencies, cutoff];

        average = mean([compressed_transcript.Confidence]);
        averages = [averages, average];

           

           % Compare baseline and compressed(part 6)
        if strcmp(transcript.Transcript, compressed_transcript.Transcript)
            disp('Transcripts Match');
        else
            disp('Transcripts Do Not Match');
            failed_transcript = compressed_transcript.Transcript;
            failed_confidence = compressed_transcript.Confidence;
            failed_cutoff = cutoff;
            break; 
        end


end

%compression ratio
compression_ratio = sum(abs(compressed_fft) > 0) / length_audio;
fprintf('Compression Ratio: %.5f\n', compression_ratio);

%cutoff frequency
fprintf('Final Cutoff Frequency: %d Hz\n', cutoff);

%Ploting average confidence vs cutoff frequencies
figure; 
plot(frequencies, averages);
xlabel('Cutoff Frequency (Hz)');
ylabel('Average Confidence');
title('Average Confidence vs. Cutoff Frequency');

%Plotting the single-sidedFourier spectrum

p2_og = abs(fft_audio/length_audio);
p1_og = p2_og(1:length_audio/2+1);
p1_og(2:end-1) = 2*p1_og(2:end-1);

f = fs*(0:(length_audio/2))/length_audio;

figure;
plot(f, p1_og);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Single-Sided Amplitude Spectrum of Original Signal');


p2_comp = abs(compressed_fft/length_audio);
p1_comp = p2_comp(1:length_audio/2 + 1);
p1_comp(2:end-1) = 2*p1_comp(2:end-1);

figure;
plot(f, p1_comp);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Single-Sided Amplitude Spectrum of Compressed Signal');


% Display the transcripts and confidences of the original and the compressed signals
disp('Original Transcript and Confidences:');
disp(transcript);
disp('Failed Compressed Transcript and Confidences:');
disp(failed_transcript);
disp(failed_confidence);

%pause(length(audioSignal) * 2); % Wait 
sound(compressed_audio, fs); % Play comp signal

%{ 
Expected
    At the higher frequencies the average confidence should be high since
    audio signal has most of the original features. In the middle this
    should gradually decrease the overall confidence should also decrease
    as higher frequencies are no longer being produced. In the lowest
    frequencies the quality of the signal would have decreased lowering 
    the confidence in the transcription. 

Observed
    There seemed to be an opposite affect where lower frequencies below
    3000 Hz should loose parts of signal. However it shows that there is
    the overal highest confidences at 3000-4000Hz. This could be because
    the speech recording of the voice sits at around or below
    3000 Hz. This way allthe external noise in the recording has been 
    filtered out. The wav2vec2.0 model may also extremely accurate and with
    the filtering of the unnecesseary frequencies this could lead to higher
    confidences. It may also be able to predict fairly well even if parts 
    of the signal are filtered out. 

    There is also a large amount of fluxuations at 4000Hz to 7000Hz,
    certain noises/signals may have taken a larger role in when more/less
    of the original signal filtered out causing a varying confidence value.
}%







