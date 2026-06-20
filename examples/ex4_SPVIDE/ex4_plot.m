% ex4_plot.m
% Plot Cauchy error and timing results for Example 4.
%
% Expected input files:
%   data/ex4_error.csv
%   data/ex4_time.csv
%
% Expected output files:
%   figures/ex4_error.pdf
%   figures/ex4_error.png
%   figures/ex4_time.pdf
%   figures/ex4_time.png
%
% This script should be placed in the same folder as ex4_run.jl,
% ex4_VoltMat.jl and ex4_GO.jl, for example:
%   examples/ex4_SPVIDE/ex4_plot.m

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

errorFile = fullfile(dataDir, 'ex4_error.csv');
timeFile  = fullfile(dataDir, 'ex4_time.csv');

if ~isfile(errorFile)
    error('Cannot find error CSV: %s', errorFile);
end
if ~isfile(timeFile)
    error('Cannot find time CSV: %s', timeFile);
end

%% Read data
Terr = readtable(errorFile);
Ttim = readtable(timeFile);

Terr = normalize_method_column(Terr);
Ttim = normalize_method_column(Ttim);

% Keep only successful rows if the status column exists.
Terr = keep_success_rows(Terr);
Ttim = keep_success_rows(Ttim);

% Sort for clean curves.
Terr = sortrows(Terr, {'method', 'N'});
Ttim = sortrows(Ttim, {'method', 'N'});

methodsErr = unique(Terr.method, 'stable');
methodsTim = unique(Ttim.method, 'stable');

%% Plot error
fig1 = figure('Color', 'w');
hold on;

for i = 1:numel(methodsErr)
    method = methodsErr(i);
    Ti = Terr(Terr.method == method, :);

    idxGrid = isfinite(Ti.N) & isfinite(Ti.cauchy_grid_error_inf) & ...
              Ti.N > 0 & Ti.cauchy_grid_error_inf > 0;
    idxCoeff = isfinite(Ti.N) & isfinite(Ti.cauchy_coeff_error_inf) & ...
               Ti.N > 0 & Ti.cauchy_coeff_error_inf > 0;

    if any(idxGrid)
        semilogy(Ti.N(idxGrid), Ti.cauchy_grid_error_inf(idxGrid), '-', ...
            'LineWidth', 1.8, ...
            'DisplayName', sprintf('%s grid Cauchy', method));
    end

    if any(idxCoeff)
        semilogy(Ti.N(idxCoeff), Ti.cauchy_coeff_error_inf(idxCoeff), '--', ...
            'LineWidth', 1.6, ...
            'DisplayName', sprintf('%s coeff Cauchy', method));
    end
end

xlabel('$N$', 'Interpreter', 'latex');
ylabel('Cauchy error', 'Interpreter', 'latex');
title('Example 4: Cauchy error versus $N$', 'Interpreter', 'latex');
legend('Location', 'southwest', 'Interpreter', 'none');
box on;

set(gca, 'XScale', 'linear', 'YScale', 'log', 'FontSize', 12, 'LineWidth', 1.0);

exportgraphics(fig1, fullfile(figDir, 'ex4_error.pdf'), 'ContentType', 'vector');
exportgraphics(fig1, fullfile(figDir, 'ex4_error.png'), 'Resolution', 300);

%% Plot time
fig2 = figure('Color', 'w');
hold on;

for i = 1:numel(methodsTim)
    method = methodsTim(i);
    Ti = Ttim(Ttim.method == method, :);

    % Prefer total_sec. If unavailable or empty, fall back to solve_pair_total_sec.
    y = [];
    yname = '';
    if ismember('total_sec', Ti.Properties.VariableNames)
        y = Ti.total_sec;
        yname = 'total sec';
    end
    if isempty(y) || all(~isfinite(y) | y <= 0)
        if ismember('solve_pair_total_sec', Ti.Properties.VariableNames)
            y = Ti.solve_pair_total_sec;
            yname = 'solve pair total sec';
        elseif ismember('solve_total_sec', Ti.Properties.VariableNames)
            y = Ti.solve_total_sec;
            yname = 'solve total sec';
        elseif ismember('solve_sec', Ti.Properties.VariableNames)
            y = Ti.solve_sec;
            yname = 'solve sec';
        else
            error('No timing column found in ex4_time.csv.');
        end
    end

    idx = isfinite(Ti.N) & isfinite(y) & Ti.N > 0 & y > 0;
    if any(idx)
        loglog(Ti.N(idx), y(idx), '-', ...
            'LineWidth', 1.8, ...
            'DisplayName', sprintf('%s %s', method, yname));
    end
end

xlabel('$N$', 'Interpreter', 'latex');
ylabel('sec', 'Interpreter', 'latex');
title('Example 4: runtime versus $N$', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'none');
box on;
grid on;
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 12, 'LineWidth', 1.0);

exportgraphics(fig2, fullfile(figDir, 'ex4_time.pdf'), 'ContentType', 'vector');
exportgraphics(fig2, fullfile(figDir, 'ex4_time.png'), 'Resolution', 300);

fprintf('Saved figures to:\n  %s\n', figDir);

%% Local helper functions
function T = normalize_method_column(T)
    if ~ismember('method', T.Properties.VariableNames)
        error('CSV file must contain a method column.');
    end

    if iscell(T.method)
        T.method = string(T.method);
    elseif iscategorical(T.method)
        T.method = string(T.method);
    elseif ischar(T.method)
        T.method = string(cellstr(T.method));
    elseif ~isstring(T.method)
        T.method = string(T.method);
    end

    T.method = strip(T.method);
end

function T = keep_success_rows(T)
    if ~ismember('status', T.Properties.VariableNames)
        return;
    end

    status = T.status;
    if iscell(status)
        status = string(status);
    elseif iscategorical(status)
        status = string(status);
    elseif ischar(status)
        status = string(cellstr(status));
    elseif ~isstring(status)
        status = string(status);
    end

    status = lower(strip(status));
    keep = status == "ok" | status == "success" | status == "done" | status == "" | ismissing(status);
    T = T(keep, :);
end
