% ex7_compare_cauchy_plot.m
% Compare Cauchy errors from collocation method and Weak_VoltMat_Cheb method.
%
% Expected input files in data/:
%   data/ex7_collocation_fast_cauchy.csv
%       or
%   data/ex7_collocation_cauchy.csv
%
%   data/ex7_weaksingular_cauchy.csv
%
% Expected output files:
%   figures/ex7_compare_cauchy_error.pdf
%   figures/ex7_compare_cauchy_error.png

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

%% Input files
collocationFastFile = fullfile(dataDir, 'ex7_collocation_fast_cauchy.csv');
collocationFileOld  = fullfile(dataDir, 'ex7_collocation_cauchy.csv');
weakFile            = fullfile(dataDir, 'ex7_weaksingular_cauchy.csv');

% Prefer the fast collocation result if it exists
if isfile(collocationFastFile)
    collocationFile = collocationFastFile;
elseif isfile(collocationFileOld)
    collocationFile = collocationFileOld;
else
    error('Cannot find collocation Cauchy CSV in data folder.');
end

if ~isfile(weakFile)
    error('Cannot find file: %s', weakFile);
end

fprintf('Using collocation file:\n  %s\n', collocationFile);
fprintf('Using coefficient method file:\n  %s\n\n', weakFile);

%% Read data
Tc = readtable(collocationFile);
Tw = readtable(weakFile);

% Each curve is cleaned independently.
[Nc_grid,  gridErrC]  = get_curve(Tc, 'grid_cauchy_inf');
[Nc_coeff, coeffErrC] = get_curve(Tc, 'coeff_cauchy_inf');

[Nw_grid,  gridErrW]  = get_curve(Tw, 'grid_cauchy_inf');
[Nw_coeff, coeffErrW] = get_curve(Tw, 'coeff_cauchy_inf');

fprintf('Valid points:\n');
fprintf('  Collocation grid error:        %d\n', numel(Nc_grid));
fprintf('  Collocation coefficient error: %d\n', numel(Nc_coeff));
fprintf('  Weak_VoltMat_Cheb grid error:  %d\n', numel(Nw_grid));
fprintf('  Weak_VoltMat_Cheb coeff error: %d\n\n', numel(Nw_coeff));

%% Plot
fig = figure('Color', 'w');
hold on;

semilogy(Nc_grid,  gridErrC,  '-',  'LineWidth', 1.8);
semilogy(Nw_grid,  gridErrW,  '-',  'LineWidth', 1.8);

semilogy(Nc_coeff, coeffErrC, '--', 'LineWidth', 1.8);
semilogy(Nw_coeff, coeffErrW, '--', 'LineWidth', 1.8);

hold off;

% Linear x-axis, logarithmic y-axis
set(gca, 'XScale', 'linear');
set(gca, 'YScale', 'log');

xlabel('$N$', 'Interpreter', 'latex');
ylabel('Cauchy error', 'Interpreter', 'latex');

legend({ ...
    'Collocation: grid error', ...
    'Weak\_VoltMat\_Cheb: grid error', ...
    'Collocation: coefficient error', ...
    'Weak\_VoltMat\_Cheb: coefficient error'}, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex');

box on;
grid on;

set(gca, 'FontSize', 13);
set(gca, 'LineWidth', 1.0);

allN = [Nc_grid(:); Nc_coeff(:); Nw_grid(:); Nw_coeff(:)];
xlim([min(allN), max(allN)]);

%% Save
pdfFile = fullfile(figDir, 'ex7_compare_cauchy_error.pdf');
pngFile = fullfile(figDir, 'ex7_compare_cauchy_error.png');

exportgraphics(fig, pdfFile, 'ContentType', 'vector');
exportgraphics(fig, pngFile, 'Resolution', 300);

fprintf('Saved:\n  %s\n  %s\n', pdfFile, pngFile);

%% Local helper
function [N, y] = get_curve(T, yname)

    if ~any(strcmp(T.Properties.VariableNames, 'N'))
        error('Missing column "N" in input CSV.');
    end

    if ~any(strcmp(T.Properties.VariableNames, yname))
        error('Missing column "%s" in input CSV.', yname);
    end

    N = T.N;
    y = T.(yname);

    ok = isfinite(N) & isfinite(y) & y > 0;

    % Use status column only if it exists, but be less fragile.
    if any(strcmp(T.Properties.VariableNames, 'status'))
        status = lower(strtrim(string(T.status)));
        okStatus = status == "ok" ...
                | status == "success" ...
                | status == "done" ...
                | status == "finished" ...
                | status == "pass";
        ok = ok & okStatus;
    end

    N = N(ok);
    y = y(ok);

    [N, idx] = sort(N);
    y = y(idx);
end