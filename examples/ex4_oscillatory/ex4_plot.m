% ex4_plot.m
%
% Plot error and timing curves for ex4_VoltMat.jl.
%
% Expected input:
%   data/ex4_VoltMat_error.csv
%   data/ex4_VoltMat_time.csv
%
% Output:
%   figures/ex4_VoltMat_error.pdf
%   figures/ex4_VoltMat_error.png
%   figures/ex4_VoltMat_time.pdf
%   figures/ex4_VoltMat_time.png

clear; clc;

%% Paths
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

dataDir = fullfile(scriptDir, 'data');
figDir  = fullfile(scriptDir, 'figures');

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

errorFile = fullfile(dataDir, 'ex4_VoltMat_error.csv');
timeFile  = fullfile(dataDir, 'ex4_VoltMat_time.csv');

if ~isfile(errorFile)
    error('Cannot find error file: %s', errorFile);
end

if ~isfile(timeFile)
    error('Cannot find time file: %s', timeFile);
end

%% Read data
E = readtable(errorFile);
T = readtable(timeFile);

%% Global plot settings
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 14);
set(groot, 'defaultLineLineWidth', 1.8);

%% Error curve
fig1 = figure('Color', 'w');
semilogy(E.N, E.err_inf, '-', 'LineWidth', 1.8);

xlabel('$N$', 'Interpreter', 'latex');
ylabel('$\|u_N-u\|_{\infty}$', 'Interpreter', 'latex');
title('Cauchy / exact error', 'Interpreter', 'latex');

xlim([min(E.N), max(E.N)]);
box on;
grid on;

set(gca, 'XScale', 'linear');
set(gca, 'YScale', 'log');

exportgraphics(fig1, fullfile(figDir, 'ex4_VoltMat_error.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig1, fullfile(figDir, 'ex4_VoltMat_error.png'), ...
    'Resolution', 300);

%% Timing curve: total time
fig2 = figure('Color', 'w');
plot(T.N, T.total_sec, '-', 'LineWidth', 1.8);

xlabel('$N$', 'Interpreter', 'latex');
ylabel('Time (sec)', 'Interpreter', 'latex');
title('Total computational time', 'Interpreter', 'latex');

xlim([min(T.N), max(T.N)]);
box on;
grid on;

set(gca, 'XScale', 'linear');

exportgraphics(fig2, fullfile(figDir, 'ex4_VoltMat_time.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig2, fullfile(figDir, 'ex4_VoltMat_time.png'), ...
    'Resolution', 300);

%% Timing components
fig3 = figure('Color', 'w');
plot(T.N, T.op_sec, '-', 'LineWidth', 1.8); hold on;
plot(T.N, T.sys_sec, '-', 'LineWidth', 1.8);
plot(T.N, T.rhs_sec, '-', 'LineWidth', 1.8);
plot(T.N, T.solve_sec, '-', 'LineWidth', 1.8);
plot(T.N, T.total_sec, '-', 'LineWidth', 2.2);

xlabel('$N$', 'Interpreter', 'latex');
ylabel('Time (sec)', 'Interpreter', 'latex');
title('Timing components', 'Interpreter', 'latex');

legend({'VoltMat assembly', 'System assembly', 'RHS', 'Solve', 'Total'}, ...
    'Interpreter', 'latex', ...
    'Location', 'northwest');

xlim([min(T.N), max(T.N)]);
box on;
grid on;

set(gca, 'XScale', 'linear');

exportgraphics(fig3, fullfile(figDir, 'ex4_VoltMat_time_components.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig3, fullfile(figDir, 'ex4_VoltMat_time_components.png'), ...
    'Resolution', 300);

fprintf('Figures saved to:\n  %s\n', figDir);