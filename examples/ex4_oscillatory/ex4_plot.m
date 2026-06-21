% plot_ex4_GO_depth_error.m
% Plot max error vs kernel depth for Float64 GO and
% BigFloat-construction-only GO.

clear; clc; close all;

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

float64File = fullfile(dataDir, 'ex4_GO_depth_sweep.csv');
bigFile     = fullfile(dataDir, 'ex4_GO_depth_sweep_bigfloat_construct_only.csv');

if ~isfile(float64File)
    error('Cannot find file: %s', float64File);
end

if ~isfile(bigFile)
    error('Cannot find file: %s', bigFile);
end

%% Read data
T64  = readtable(float64File);
Tbig = readtable(bigFile);

% Keep successful rows if status column exists
if ismember('status', T64.Properties.VariableNames)
    T64 = T64(strcmp(string(T64.status), "ok"), :);
end

if ismember('status', Tbig.Properties.VariableNames)
    Tbig = Tbig(strcmp(string(Tbig.status), "ok"), :);
end

%% Split BigFloat construction-only data
T128 = Tbig(Tbig.precision_bits == 128, :);
T256 = Tbig(Tbig.precision_bits == 256, :);

%% Sort by depth
T64  = sortrows(T64,  'depth');
T128 = sortrows(T128, 'depth');
T256 = sortrows(T256, 'depth');

%% Plot
fig = figure('Color', 'w');
hold on;

semilogy(T64.depth,  T64.max_error,  '-',  'LineWidth', 1.8);
semilogy(T128.depth, T128.max_error, '--', 'LineWidth', 1.8);
semilogy(T256.depth, T256.max_error, '-.', 'LineWidth', 1.8);

xlabel('Kernel expansion depth', 'Interpreter', 'latex');
ylabel('Maximum error', 'Interpreter', 'latex');

legend( ...
    'Float64 construction', ...
    'BigFloat 128-bit construction + Float64 solve', ...
    'BigFloat 256-bit construction + Float64 solve', ...
    'Location', 'best', ...
    'Interpreter', 'latex' ...
);

set(gca, ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex', ...
    'YScale', 'log' ...
);

grid on;
box on;

title('GO depth sweep for oscillatory Volterra equation', ...
    'Interpreter', 'latex');

%% Save
pdfFile = fullfile(figDir, 'ex4_GO_depth_max_error.pdf');
pngFile = fullfile(figDir, 'ex4_GO_depth_max_error.png');

exportgraphics(fig, pdfFile, 'ContentType', 'vector');
exportgraphics(fig, pngFile, 'Resolution', 300);

fprintf('Saved figure:\n  %s\n  %s\n', pdfFile, pngFile);

%% Print best depth information
[err64, idx64] = min(T64.max_error);
[err128, idx128] = min(T128.max_error);
[err256, idx256] = min(T256.max_error);

fprintf('\nBest errors:\n');
fprintf('  Float64 construction: depth = %d, max_error = %.6e\n', ...
    T64.depth(idx64), err64);
fprintf('  BigFloat 128-bit construction + Float64 solve: depth = %d, max_error = %.6e\n', ...
    T128.depth(idx128), err128);
fprintf('  BigFloat 256-bit construction + Float64 solve: depth = %d, max_error = %.6e\n', ...
    T256.depth(idx256), err256);