var longitude=-110.75621475219727
var latitude=33.50277070790696
var point = ee.Geometry.Point(longitude, latitude);
var lonLat = ee.Image.pixelLonLat().sample(point, 30).first();
var utmZone = lonLat.get('UTM zone');
var image = ee.Image('LANDSAT/LC08/C01/T1/LC08_036037_20200817');

// Print the UTM zone number
print('UTM Zone:', utmZone);

// Get the UTM zone of the point
var lon = point.coordinates().get(0);
var utmZone = ee.Number(lon).divide(6).add(31).floor().int();

// Print the UTM zone
print('UTM Zone:', utmZone);

// Define a variable
var condition = latitude>0;

// If-else statement
var string = ee.Algorithms.If(condition,
  ee.String('EPSG:326'),// north 
  ee.String('EPSG:327')); // south
var string=ee.String(string)
// Print the result
print('Result:', string);


// Concatenate the strings
var concatenatedString = string.cat(utmZone);

// Print the concatenated string. This is the UTM EPSG code
print('Concatenated String:', concatenatedString);

// Create a projection object for UTM Zone 52N
var utmProjection = ee.Projection(concatenatedString.getInfo());
var wgsProjection =ee.Projection('EPSG:4326');
// Convert point coordinate to UTM Zone 52N
var utmPoint = point.transform(utmProjection);

// Print the UTM coordinates
print('UTM Point:', utmPoint);

var x_utm=utmPoint.coordinates().get(0)
var y_utm=utmPoint.coordinates().get(1)

print('Type of myVariable:', typeof a);


var b=ee.Number(66)
print('Type of myVariable b:', typeof b);

var xmin=x_utm.getInfo()
var ymin=y_utm.getInfo()
var xmax=xmin+30*256 // output is 256xx256
var ymax=ymin+30*256
print('Type of myVariable:', typeof xmin);

var roi_ = ee.Geometry.Rectangle([xmin,ymin,xmax,ymax],utmProjection);

var upleft=ee.Geometry.Point([xmin,ymin],utmProjection) //get up left corner in UTM
var upleft_wgs=upleft.transform(wgsProjection) // convert from utm to wgs
var lowright=ee.Geometry.Point([xmax,ymax],utmProjection)
var lowright_wgs=lowright.transform(wgsProjection)

var roi_wgs = ee.Geometry.Rectangle(
  [
    upleft_wgs,  // min x and y
    lowright_wgs   // max x and y
  ]
);


print('upleft_wgs')
print(upleft_wgs)
print('lowright_wgs')
print(lowright_wgs)


var visualization = {
  bands: ['B7', 'B6', 'B2'],
  min: 0.0,
  max: 35535,
};
// Display the selected image
Map.addLayer(image, visualization, 'Selected Image');
var wgsProjection = ee.Projection('EPSG:4326');

//var roi_wgs84=roi_.transform(wgsProjection);
Map.addLayer(roi_wgs, {}, 'rectangleBounds');

Map.centerObject(point, 15);

var clippedImage = image.clip(roi_wgs);
print(clippedImage)
Map.addLayer(clippedImage, visualization, 'subset Image');

var selectedBands = clippedImage.select(['B1', 'B2', 'B3','B4', 'B5', 'B6','B7', 'B9', 'B10','B11']); // Example: select bands 4 (Red), 3 (Green), and 2 (Blue)
print(concatenatedString.getInfo())
print('Type of myVariable concatenatedString:', typeof concatenatedString);

// Export the selected bands to Google Drive with specified pixel size
Export.image.toDrive({
  image: selectedBands,
  description: 'Selected_Bands',
  folder: 'MyFolder',
  scale: 30, // Adjust the scale according to your desired resolution
  region: clippedImage.geometry(),
  crs: concatenatedString.getInfo(),
});

// Export the selected bands to Google Drive with specified image dimension
Export.image.toDrive({
  image: selectedBands,
  description: 'Selected_Bands_rows_coms',
  folder: 'MyFolder',
  region: clippedImage.geometry(),
  crs: concatenatedString.getInfo(),
  dimensions:"256x256",
});
