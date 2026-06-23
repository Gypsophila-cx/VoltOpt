clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'data');
figDir = fullfile(scriptDir, 'figures');

if ~isfolder(figDir)
    mkdir(figDir);
end

csvfile = fullfile(dataDir, 'ex5_volterra_schur_besselj1_pseudospectrum.csv');
if ~isfile(csvfile)
    error('Cannot find CSV file: %s', csvfile);
end

T = readtable(csvfile);

x = unique(T.x, "sorted");
y = unique(T.y, "sorted");

nx = numel(x);
ny = numel(y);

Z = reshape(T.log10_pse, [ny, nx]).';

[X, Y] = meshgrid(x, y);
Zplot = Z.';

levels = -14:1:-1;

figure("Color", "w");

contour(X, Y, Zplot, levels, ...
    "LineWidth", 1.5);

colormap(turbo);

cb = colorbar;
cb.Label.Interpreter = "latex";
cb.Label.String = "$\log_{10}(\epsilon)$";

clim([min(levels), max(levels)]);

xlabel("$\mathrm{Re}(z)$", "Interpreter", "latex");
ylabel("$\mathrm{Im}(z)$", "Interpreter", "latex");
title("Pseudospectrum", "Interpreter", "latex");

axis equal tight;
box on;
set(gca, "FontSize", 12, "LineWidth", 1.0);

outpdf = fullfile(figDir, "ex5_pseudospectrum.pdf");
outpng = fullfile(figDir, "ex5_pseudospectrum.png");

exportgraphics(gcf, outpdf, ...
    "ContentType", "vector", ...
    "BackgroundColor", "white");
exportgraphics(gcf, outpng, ...
    "Resolution", 300, ...
    "BackgroundColor", "white");

fprintf('Figure saved to:\n  %s\n  %s\n', outpdf, outpng);
