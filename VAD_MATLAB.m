function [ Y ] = VAD_MATLAB( filename)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[y,fs,~,~] = readwav(filename,'r');

% Resampling
old_frequency = fs;
new_frequency = 16000;

y = round(v_resample(y,new_frequency,old_frequency));

% Voice activity detector
[decision,~]=vadsohn(y,new_frequency,'a'); % Return the decision vector

% Make the signal and the decision vector of the same(y and vs1 might 
% not be from the same length)
y_signal = y(1:length(decision));
y_new = y_signal .* decision;

% Final output by eliminating zeros
Y = y_new(y_new~=0);

end

