# Location History JSON Converter

This Python script takes the JSON file of your google maps location history and converts it into other
formats.

You can export the encrypted location history from Android devices as
described
[here](https://support.google.com/maps/thread/280205453/how-do-i-download-my-timeline-history).

You used to be able to get the location history stored on google servers
via [Google Takeout](https://takeout.google.com/settings/takeout/custom/location_history).
The files exported here can still be converted.

### Requirements

*  [Install python](https://wiki.python.org/moin/BeginnersGuide/Download) (3.2+) if you don't have it installed already.

*  Download the python script by either cloning this repository
   (`git clone https://github.com/Scarygami/location-history-json-converter`)
   or [downloading the script file](https://raw.githubusercontent.com/Scarygami/location-history-json-converter/master/location_history_json_converter.py).

*  Install dependencies using `pip install -r requirements.txt`

   See below if you encounter issues installing the Shapely package

*  Export the encrypted location history from Android devices as described [here](https://support.google.com/maps/thread/280205453/how-do-i-download-my-timeline-history).

   I find it easiest to place the exported json file in the same folder where the script is located.

### Usage
```
python location_history_json_converter.py [-h]
                                          [-f {kml,json,js,jsonfull,jsfull,csv,csvfull,csvfullest,gpx,gpxtracks}]
                                          [-i] [-s STARTDATE] [-e ENDDATE] [--starttime STARTTIME]
                                          [--endtime ENDTIME] [-a ACCURACY] [-c] [-l] [-v VARIABLE]
                                          [--separator SEPARATOR] [-p [lat,lon ...]]
                                          input output

positional arguments:
  input                 Input File (Location History.json)
  output                Output File (will be overwritten!)

options:
  -h, --help            show this help message and exit
  -f {kml,json,js,jsonfull,jsfull,csv,csvfull,csvfullest,gpx,gpxtracks}, --format {kml,json,js,jsonfull,jsfull,csv,csvfull,csvfullest,gpx,gpxtracks}
                        Format of the output
  -i, --iterative       Loads the JSON file iteratively, to be able to handle bigger files
  -s STARTDATE, --startdate STARTDATE
                        The Start Date - format YYYY-MM-DD (defaults to 0h00m)
  -e ENDDATE, --enddate ENDDATE
                        The End Date - format YYYY-MM-DD (defaults to 23h59m59s)
  --starttime STARTTIME
                        The Start Time - format HH:MM, only used if Start Date is set
  --endtime ENDTIME     The End Time - format HH:MM, only used if End Date is set
  -a ACCURACY, --accuracy ACCURACY
                        Maximum accuracy (in meters), lower is better.
  -c, --chronological   Sort items in chronological order (might be unnessary)
  -l, --legacy          Use the legacy data in exports from the mobile device (no effect on data
                        exported from servers)
  -v VARIABLE, --variable VARIABLE
                        Variable name to be used for js output
  --separator SEPARATOR
                        Separator to be used for CSV formats, defaults to comma
  -p [lat,lon ...], --polygon [lat,lon ...]
                        List of points (lat, lon) that create a polygon. If two points are given a
                        rectangle is created.
```

### Special requirements for some options

#### `-i, --iterative`

The iterative parsing mode is achieved using the [ijson](https://pypi.org/project/ijson/).

To be able to use this option you will have to install it with

    pip install ijson

#### `-p, --polygon`

Using this option you can specify a list of coordinates to define a polygon,
and only locations that are in this polygon will be added to the output file.

E.g `-p 43.665,10.334 43.815,10.492` to only include locations in the rectangle
defined by the two corner points.

If you have negative latitudes you will need to but the coordinate in quotes
with an extra space before the minus sign, so that `argparse` can detect and read
the arguments correctly.

    --polygon 20,-70 " -20,-50"

The polygon filtering is achieved using [Shapely](https://pypi.org/project/Shapely/).

To be able to use this option you will have to install it with

    pip install Shapely

On Windows this command will most likely fail. Instead you can download a wheel
that matches your OS and Python Version from https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely

You can then install Shapely using this command:

    python -m pip install Shapely-X-cpX-cpXm-winX.whl


### Available formats

#### kml (default)
KML file with placemarks for each location in your Location History.
Each placemark will have a location, a timestamp, and accuracy/speed/altitude as available.
Data produced is valid KML 2.2.

#### csv
Comma-separated text file with a timestamp field and a location field, suitable for upload to Fusion Tables.

#### csvfull
Comma-separated text file with all location information, excluding activities

#### csvfullest
Comma-separated text file with all location information, including activities

#### json
Smaller JSON file with only the timestamp and the location.

#### js
JavaScript file which sets a variable in global namespace (default: window.locationJsonData)
to the full data object for easy access in local scripts.
Just include the js file before your actual script.
Only timestamp and location are included.

#### jsonfull, jsfull
These types essentially make a full copy of the entries in the original JSON File in json or js format.
With the option of filtering start and end date this can be used to create a smaller file in iterative mode,
that can then be handled without iterative mode (necessary for gpxtracks and the chronological option).

#### gpx
GPS Exchange Format including location, timestamp, and accuracy/speed/altitude as available.
Data produced is valid GPX 1.1.  Points are stored as individual, unrelated waypoints (like the other formats, except for gpxtracks).

#### gpxtracks
GPS Exchange Format including location, timestamp, and accuracy/speed/altitude as available.
Data produced is valid GPX 1.1.  Points are grouped together into tracks by time and location (specifically, two chronological points split a track if they differ by over 10 minutes or approximately 40 kilometers).
