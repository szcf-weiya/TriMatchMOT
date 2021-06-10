using PyCall
using Images
using ImageView
using ImageDraw
using Gtk.ShortNames
using Cairo
# import local module
# do not forget to add sys path: https://github.com/JuliaPy/PyCall.jl/issues/48
py"""
import sys
sys.path.insert(0, "../detection")
import video
# https://www.geeksforgeeks.org/reloading-modules-python/
# and https://stackoverflow.com/questions/5516783/how-to-reload-python-module-imported-using-from-module-import
import importlib
importlib.reload(video)
from video import get_centers, match_video, match_video2
"""

includet("../DP/dp.jl")
# read all images
function load_imgs()
    raw_coors, raw_ims, raw_masks = py"match_video"(30, area_min = 166.75, start = 0)
    arr_imgs = [colorview(RGB, permutedims(reinterpret.(N0f8, im), (3, 1, 2))) for im in raw_ims]
    arr_masks = [Gray.(reinterpret.(N0f8, mask)) for mask in raw_masks]
    imgs = cat(arr_imgs..., dims = 3)
    masks = cat(arr_masks..., dims = 3)
    X = [[coor[j,:] .+ 1 for j = 1:size(coor, 1)] for coor in raw_coors]
end

function run_match()
    # match by tripartite matching
    σ = 5
    Ms = bottom_up_match_optim2(X, method="prior_max", σ = σ, δ = 0, robust = 1, history = false);
    # matching by min-cost flow
    ms = match_by_mincost(X, method2=true, method="distance", maxdistance = 2500)
end
