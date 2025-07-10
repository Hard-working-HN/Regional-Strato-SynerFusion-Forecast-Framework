function [amplitude_envelope, instantaneous_frequency] = hua_baoluo(y, fs, style, varargin)

    y_hht = hilbert(y); 
    amplitude_envelope = abs(y_hht); 
    amplitude_envelope = amplitude_envelope - mean(amplitude_envelope); 

    instantaneous_phase = angle(y_hht);
    instantaneous_frequency = (fs / (2 * pi)) * [0, diff(instantaneous_phase)]; 

    if nargin == 3
        hua_fft(amplitude_envelope, fs, style);
    elseif nargin == 5
        f1 = varargin{1};
        f2 = varargin{2};
        hua_fft(amplitude_envelope, fs, style, f1, f2);
    else
        error('调用函数的输入参数数目不正确，输入参数只能是三个或者五个');
    end
end
