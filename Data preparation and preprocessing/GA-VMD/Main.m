clc
clear all
close all
warning('off', 'all');
tic 

if isempty(gcp('nocreate'))
    parpool('local'); 
end

load("train.mat") 
s = signal';
signal = s;
fs = 160; 

maxgen = 25;  
sizepop = 20; 
pcross = 0.8;  
pmutation = 0.1; 
nvar = 2; 
lenchrom = ones(1, nvar); 

bound = [100 3000; 4 11];


chrom = zeros(sizepop, nvar);
fitness = zeros(sizepop, 1);


parfor i = 1:sizepop
    chrom(i,:) = Code(lenchrom, bound);   
    x = chrom(i,:);
    fitness(i) = objfun(x, s);
end

individuals.chrom = chrom;
individuals.fitness = fitness;

[bestfitness(1), bestindex] = min(individuals.fitness);  
bestchrom(1,:) = individuals.chrom(bestindex,:); 
avgfitness(1) = sum(individuals.fitness) / sizepop; 

start_time_train = cputime;
for gen = 1:maxgen
    disp(['迭代次数：', num2str(gen)])
    individuals = Select(individuals, sizepop); 

    chrom = individuals.chrom;
    fitness = individuals.fitness;

    avgfitness(gen + 1) = sum(fitness) / sizepop;

    chrom = Cross(pcross, lenchrom, chrom, sizepop, bound);

    chrom = Mutation(pmutation, lenchrom, chrom, sizepop, gen, maxgen, bound);

    fitness = zeros(sizepop, 1); 
    parfor j = 1:sizepop
        x = chrom(j,:); 
        fitness(j) = objfun(x, s);
    end

    individuals.chrom = chrom;
    individuals.fitness = fitness;

    [newbestfitness, newbestindex] = min(fitness);
    [worestfitness, worestindex] = max(fitness);

    if bestfitness(gen) > newbestfitness
        bestfitness(gen + 1) = newbestfitness;
        bestchrom(gen + 1,:) = chrom(newbestindex,:);
    else
        bestfitness(gen + 1) = bestfitness(gen);
        bestchrom(gen + 1,:) = bestchrom(gen,:);
    end
    chrom(worestindex,:) = bestchrom(gen + 1,:);
    fitness(worestindex) = bestfitness(gen + 1);

    individuals.chrom = chrom;
    individuals.fitness = fitness;

    avgfitness(gen + 1) = sum(fitness) / sizepop;
end

figure

set(0,'defaultAxesFontName', 'Monospaced');
set(0,'defaultAxesFontSize', 10);

plot(0:maxgen, bestfitness,'r-o','linewidth',2)
xlabel('迭代次数')
ylabel('适应度')
legend('最佳适应度')
axis tight
grid on

[final_bestfitness, final_gen] = min(bestfitness);
optimal_chrom = bestchrom(final_gen,:);
alpha = round(optimal_chrom(1)); 
K = round(optimal_chrom(2));  
tau = 0;           
DC = 0;             
init = 1;          
tol = 1e-6;

[u, u_hat, omega] = VMD(s, alpha, tau, K, DC, init, tol);

figure; 
for k = 1:K
    subplot(K+2,1,k);
    plot(u(k,:),'b');
    title(sprintf('VMD 分解 IMF %d 分量时域波形', k));
end
subplot(K+2,1,K+1);
plot(signal,'b');
title('原始信号时域波形');
reconstructed_signal = sum(u,1);
subplot(K+2,1,K+2);
plot(reconstructed_signal,'b');
title('重构信号时域波形');

amplitude_envelopes = zeros(K, length(u));

parfor k = 1:K
    imf = u(k, :);

    imf_hilbert = hilbert(imf);
    envelope = abs(imf_hilbert);


    amplitude_envelopes(k, :) = envelope;
end

filename = 'VMD_train_results.xlsx';

if exist(filename, 'file') == 2
    delete(filename); 
end

Original_Signal = array2table(signal');
writetable(Original_Signal, filename, 'Sheet', 'Original_Signal', 'WriteVariableNames', false);

IMF_TimeDomain = array2table(u');
writetable(IMF_TimeDomain, filename, 'Sheet', 'IMF_TimeDomain', 'WriteVariableNames', false);

IMF_Spectra = array2table(u_hat');
writetable(IMF_Spectra, filename, 'Sheet', 'IMF_Spectra', 'WriteVariableNames', false);

IMF_Envelopes = array2table(amplitude_envelopes');
writetable(IMF_Envelopes, filename, 'Sheet', 'IMF_Envelopes', 'WriteVariableNames', false);

Frequencies = array2table(omega');
writetable(Frequencies, filename, 'Sheet', 'Frequencies', 'WriteVariableNames', false);

vmd_params = table({'alpha'; 'K'; 'tau'; 'DC'; 'init'; 'tol'}, [alpha; K; tau; DC; init; tol], 'VariableNames', {'Parameter', 'Value'});
writetable(vmd_params, filename, 'Sheet', 'VMD_Parameters');

Generation = (0:maxgen)';
BestFitness = bestfitness(:);

disp(['Size of Generation: ', num2str(size(Generation, 1)), ' x ', num2str(size(Generation, 2))]);
disp(['Size of BestFitness: ', num2str(size(BestFitness, 1)), ' x ', num2str(size(BestFitness, 2))]);
if size(Generation, 1) ~= size(BestFitness, 1)
    error('Generation 和 BestFitness 的长度不一致，无法创建表格。');
end
GA_Evolution_Trace = table(Generation, BestFitness, 'VariableNames', {'Generation', 'BestFitness'});
writetable(GA_Evolution_Trace, filename, 'Sheet', 'GA_Evolution_Trace');

ga_params = table({'maxgen'; 'sizepop'; 'pcross'; 'pmutation'; 'nvar'}, [maxgen; sizepop; pcross; pmutation; nvar], 'VariableNames', {'Parameter', 'Value'});
writetable(ga_params, filename, 'Sheet', 'GA_Parameters');

disp(['所有分解结果已保存到 ', filename]);

toc 

delete(gcp('nocreate'));