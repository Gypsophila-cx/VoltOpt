% ex4_plot.m
% Plot max error curves for GO and VoltMat depth sweeps, N = 200.
%
% Expected input files:
%   data/ex4_GO_depth_sweep_N200.csv
%   data/ex4_VoltMat_depth_sweep_N200.csv
%
% Output files:
%   figures/ex4_depth_error_N200.pdf
%   figures/ex4_depth_error_N200.png

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

goFile      = fullfile(dataDir, 'ex4_GO_depth_sweep_N200.csv');
voltmatFile = fullfile(dataDir, 'ex4_VoltMat_depth_sweep_N200.csv');

if ~isfile(goFile)
    error('Cannot find file: %s', goFile);
end

if ~isfile(voltmatFile)
    error('Cannot find file: %s', voltmatFile);
end

%% Read data
Tgo = readtable(goFile);
Tvm = readtable(voltmatFile);

%% Keep successful rows
if ismember('status', Tgo.Properties.VariableNames)
    statusGO = lower(string(Tgo.status));
    Tgo = Tgo(statusGO == "ok" | statusGO == "success", :);
end

if ismember('status', Tvm.Properties.VariableNames)
    statusVM = lower(string(Tvm.status));
    Tvm = Tvm(statusVM == "ok" | statusVM == "success", :);
end

%% Remove invalid rows
Tgo = Tgo(isfinite(Tgo.depth) & isfinite(Tgo.max_error) & Tgo.max_error > 0, :);
Tvm = Tvm(isfinite(Tvm.depth) & isfinite(Tvm.max_error) & Tvm.max_error > 0, :);

%% Sort by depth
Tgo = sortrows(Tgo, 'depth');
Tvm = sortrows(Tvm, 'depth');

%% Plot max error
fig = figure('Color', 'w');
hold on;

semilogy(Tgo.depth, Tgo.max_error, ...
    'LineWidth', 1.8);

semilogy(Tvm.depth, Tvm.max_error, ...
    'LineWidth', 1.8);

xlabel('kernel expansion depth', 'Interpreter', 'latex');
ylabel('error', 'Interpreter', 'latex');

legend( ...
    'Gutleb-Olver', ...
    'VoltMat-Cheb', ...
    'Location', 'best', ...
    'Interpreter', 'latex' ...
);

set(gca, ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex', ...
    'XScale', 'linear', ...
    'YScale', 'log' ...
);

box on;

%% Save figures
pdfFile = fullfile(figDir, 'ex4_depth_error_N200.pdf');
pngFile = fullfile(figDir, 'ex4_depth_error_N200.png');

exportgraphics(fig, pdfFile, 'ContentType', 'vector');
exportgraphics(fig, pngFile, 'Resolution', 300);

fprintf('Saved figures:\n');
fprintf('  %s\n', pdfFile);
fprintf('  %s\n', pngFile);

%% Print best errors
[goBestErr, goIdx] = min(Tgo.max_error);
[vmBestErr, vmIdx] = min(Tvm.max_error);

fprintf('\nBest errors:\n');
fprintf('  GO sparse Volterra, N=200: depth = %d, max_error = %.6e\n', ...
    Tgo.depth(goIdx), goBestErr);
fprintf('  VoltMat-Cheb,       N=200: depth = %d, max_error = %.6e\n', ...
    Tvm.depth(vmIdx), vmBestErr);

%% Optional diagnostics for VoltMat
if ismember('exact_residual_inf', Tvm.Properties.VariableNames)
    [bestExactRes, idxRes] = min(Tvm.exact_residual_inf);
    fprintf('\nVoltMat exact residual diagnostic:\n');
    fprintf('  best exact_residual_inf: depth = %d, exact_residual_inf = %.6e\n', ...
        Tvm.depth(idxRes), bestExactRes);
end

if ismember('rhs_tail_inf', Tvm.Properties.VariableNames)
    [bestTail, idxTail] = min(Tvm.rhs_tail_inf);
    fprintf('  best rhs_tail_inf:       depth = %d, rhs_tail_inf = %.6e\n', ...
        Tvm.depth(idxTail), bestTail);
end