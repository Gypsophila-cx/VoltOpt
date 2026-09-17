% ex2_plot.m
%
% Purpose:
%   Plot Example 2 error, solve time, and build-time comparisons.
%
% Problem:
%   Aging Maxwell second-kind Volterra equation solved by the VoltMat and
%   Gutleb--Olver scripts.
%
% Input files:
%   data/ex2_VoltMat_error.csv
%   data/ex2_VoltMat_time.csv
%   data/ex2_GO_error.csv
%   data/ex2_GO_time.csv
%
% Output files:
%   figures/ex2_error.pdf
%   figures/ex2_error.png
%   figures/ex2_time.pdf
%   figures/ex2_time.png
%   figures/ex2_build_time.pdf
%   figures/ex2_build_time.png
%
% Run:
%   matlab -batch "run('examples/ex2_2ndVolterraEqn/ex2_plot.m')"

clear;
clc;
close all;

% ============================================================
% Paths
% Put this script in examples/ex2_2ndVolterraEqn
% ============================================================

this_dir = fileparts(mfilename('fullpath'));

data_dir = fullfile(this_dir, 'data');
fig_dir  = fullfile(this_dir, 'figures');

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

csv_volt_error = fullfile(data_dir, 'ex2_VoltMat_error.csv');
csv_go_error   = fullfile(data_dir, 'ex2_GO_error.csv');

csv_volt_time = fullfile(data_dir, 'ex2_VoltMat_time.csv');
csv_go_time   = fullfile(data_dir, 'ex2_GO_time.csv');

% ============================================================
% Read CSV files
% ============================================================

T_volt_error = readtable(csv_volt_error);
T_go_error   = readtable(csv_go_error);

T_volt_time = readtable(csv_volt_time);
T_go_time   = readtable(csv_go_time);

% Keep successful rows if status column exists
T_volt_error = keep_ok_rows(T_volt_error);
T_go_error   = keep_ok_rows(T_go_error);

T_volt_time = keep_ok_rows(T_volt_time);
T_go_time   = keep_ok_rows(T_go_time);

% Sort by N
T_volt_error = sortrows(T_volt_error, 'N');
T_go_error   = sortrows(T_go_error, 'N');

T_volt_time = sortrows(T_volt_time, 'N');
T_go_time   = sortrows(T_go_time, 'N');

% ============================================================
% Column extraction
% ============================================================

N_volt_error = T_volt_error.N;
N_go_error   = T_go_error.N;

N_volt_time = T_volt_time.N;
N_go_time   = T_go_time.N;

err_volt = T_volt_error.relative_error;
err_go   = T_go_error.relative_error;

time_volt = get_column(T_volt_time, ...
    {'end2end_median_sec', 'end_to_end_median_sec'});
time_go = get_column(T_go_time, ...
    {'end2end_median_sec', 'end_to_end_median_sec'});

build_volt = get_column(T_volt_time, ...
    {'build_median_sec', 'operator_build_median_sec'});
build_go = get_column(T_go_time, ...
    {'build_median_sec', 'operator_build_median_sec'});

% Remove nonpositive or nonfinite values before log plots
idx_volt_err = ...
    isfinite(N_volt_error) & ...
    isfinite(err_volt) & ...
    err_volt > 0;

idx_go_err = ...
    isfinite(N_go_error) & ...
    isfinite(err_go) & ...
    err_go > 0;

idx_volt_time = ...
    isfinite(N_volt_time) & ...
    isfinite(time_volt) & ...
    time_volt > 0;

idx_go_time = ...
    isfinite(N_go_time) & ...
    isfinite(time_go) & ...
    time_go > 0;

idx_volt_build = ...
    isfinite(N_volt_time) & ...
    isfinite(build_volt) & ...
    build_volt > 0;

idx_go_build = ...
    isfinite(N_go_time) & ...
    isfinite(build_go) & ...
    build_go > 0;

% ============================================================
% Plot 1: error vs N
% Error CSV uses linearly spaced N
% ============================================================

figure('Color', 'w');

semilogy( ...
    N_volt_error(idx_volt_err), ...
    err_volt(idx_volt_err), ...
    '-', ...
    'LineWidth', 1.5);

hold on;

semilogy( ...
    N_go_error(idx_go_err), ...
    err_go(idx_go_err), ...
    '-', ...
    'LineWidth', 1.5);

xlabel('$N$', ...
    'Interpreter', 'latex');

ylabel('error', ...
    'Interpreter', 'latex');

legend( ...
    {'Duffy--Chebyshev', 'Gutleb--Olver'}, ...
    'Interpreter', 'latex', ...
    'Location', 'southwest');

ax = gca;

set(ax, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12);

box on;

% ============================================================
% Preserve MATLAB's default x-axis exponent position
% ============================================================
%
% MATLAB normally draws the x-axis multiplier "x10^4" using the
% ruler's SecondaryLabel.  In some MATLAB versions this object is
% not included correctly by exportgraphics.
%
% We first let MATLAB construct the exponent exactly as usual,
% record its default location and formatting, then replace it by
% an ordinary text object at the same location.
% ============================================================

% Force the same exponent that MATLAB selects automatically here.
ax.XAxis.Exponent = 4;

drawnow;

% Preserve MATLAB's automatically selected ticks.
defaultXTicks = ax.XTick;

% Access MATLAB's native exponent label.
hExponent = ax.XAxis.SecondaryLabel;

% Express its location relative to the axes.
hExponent.Units = 'normalized';

% Save MATLAB's own default placement and formatting.
exponentPosition = hExponent.Position;
exponentHorizontalAlignment = hExponent.HorizontalAlignment;
exponentVerticalAlignment   = hExponent.VerticalAlignment;
exponentRotation            = hExponent.Rotation;
exponentFontName            = hExponent.FontName;
exponentFontSize            = hExponent.FontSize;
exponentFontWeight          = hExponent.FontWeight;

% Disable MATLAB's SecondaryLabel exponent.
ax.XAxis.Exponent = 0;

% Restore the same physical tick locations, but explicitly display
% them in units of 10^4.
ax.XTick = defaultXTicks;
ax.XTickLabel = compose('%g', defaultXTicks / 1e4);

drawnow;

% Reproduce the exponent at MATLAB's own default position.
%
% This is now an ordinary text object, so exportgraphics includes
% it reliably in PDF and PNG output.
text( ...
    ax, ...
    exponentPosition(1), ...
    exponentPosition(2), ...
    '$\times 10^{4}$', ...
    'Units', 'normalized', ...
    'Interpreter', 'latex', ...
    'HorizontalAlignment', exponentHorizontalAlignment, ...
    'VerticalAlignment', exponentVerticalAlignment, ...
    'Rotation', exponentRotation, ...
    'FontName', exponentFontName, ...
    'FontSize', exponentFontSize, ...
    'FontWeight', exponentFontWeight, ...
    'Clipping', 'off', ...
    'HandleVisibility', 'off');

drawnow;

exportgraphics( ...
    gcf, ...
    fullfile(fig_dir, 'ex2_error.pdf'), ...
    'ContentType', 'vector');

exportgraphics( ...
    gcf, ...
    fullfile(fig_dir, 'ex2_error.png'), ...
    'Resolution', 300);

% ============================================================
% Plot 2: end-to-end median time vs N
% Time CSV uses logarithmically spaced N
% ============================================================

figure('Color', 'w');

loglog( ...
    N_volt_time(idx_volt_time), ...
    time_volt(idx_volt_time), ...
    '-', ...
    'LineWidth', 1.5);

hold on;

loglog( ...
    N_go_time(idx_go_time), ...
    time_go(idx_go_time), ...
    '-', ...
    'LineWidth', 1.5);

xlabel('$N$', ...
    'Interpreter', 'latex');

ylabel('time (sec)', ...
    'Interpreter', 'latex');

legend( ...
    {'Duffy--Chebyshev', 'Gutleb--Olver'}, ...
    'Interpreter', 'latex', ...
    'Location', 'northwest');

axis([1e2, 3e4, 1e-3, 1e2]);

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12);

box on;

exportgraphics( ...
    gcf, ...
    fullfile(fig_dir, 'ex2_time.pdf'), ...
    'ContentType', 'vector');

exportgraphics( ...
    gcf, ...
    fullfile(fig_dir, 'ex2_time.png'), ...
    'Resolution', 300);

% ============================================================
% Plot 3: operator build median time vs N
% Time CSV uses logarithmically spaced N
% ============================================================

figure('Color', 'w');

loglog( ...
    N_volt_time(idx_volt_build), ...
    build_volt(idx_volt_build), ...
    '-', ...
    'LineWidth', 1.5);

hold on;

loglog( ...
    N_go_time(idx_go_build), ...
    build_go(idx_go_build), ...
    '-', ...
    'LineWidth', 1.5);

xlabel('$N$', ...
    'Interpreter', 'latex');

ylabel('build time (sec)', ...
    'Interpreter', 'latex');

legend( ...
    {'Duffy--Chebyshev', 'Gutleb--Olver'}, ...
    'Interpreter', 'latex', ...
    'Location', 'northwest');

axis([1e2, 3e4, 1e-3, 1e2]);

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12);

box on;

exportgraphics( ...
    gcf, ...
    fullfile(fig_dir, 'ex2_build_time.pdf'), ...
    'ContentType', 'vector');

exportgraphics( ...
    gcf, ...
    fullfile(fig_dir, 'ex2_build_time.png'), ...
    'Resolution', 300);

fprintf('Figures saved to:\n%s\n', fig_dir);

% ============================================================
% Helper functions
% ============================================================

function T = keep_ok_rows(T)

    if ismember('status', T.Properties.VariableNames)
        T = T(strcmp(string(T.status), "ok"), :);
    end

end

function x = get_column(T, names)

    vars = T.Properties.VariableNames;

    for k = 1:numel(names)

        name = names{k};

        if ismember(name, vars)
            x = T.(name);
            return;
        end

    end

    error( ...
        'None of the requested columns exist: %s', ...
        strjoin(names, ', '));

end