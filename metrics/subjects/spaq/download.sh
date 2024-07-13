L=https://titan.gml-team.ru:5003/fsdownload/TRm4ciD3T/BL_release.pt
wget --backups=1 -nv "$L" "$L"; rm "$(basename "$L").1"
