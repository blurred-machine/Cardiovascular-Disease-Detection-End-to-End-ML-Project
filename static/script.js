$(document).ready(function() {

	function StatsProcessor() {	
		var newStats = new Stats();
		//Call Methods
		newStats.init();	
		newStats.processLine();	
		newStats.collectStats();
		//Update View
		updateDom();
	};

	//Create Stats class
	var Stats = function(mergedData,dataCount,dataMean,dataMode,longestRow){
		this.mergedData = mergedData;
		this.dataCount = dataCount;
		this.dataMean = dataMean;	
		this.dataMode = dataMode;
		this.longestRow = longestRow;	
	};
	
	// init method - prepares data for processing, sets all Stats properties to default values
	Stats.prototype.init = function(){

		// Format data
		for (i = 0; i < data.length; i++) {		
			// Remove non-numerical data from array
			data[i] = data[i].filter(Number)		
			// Convert data to integer
			data[i] = data[i].map(function (x) { 
				return parseInt(x, 10); 
			});
		}
		
		//Set default values
		mergedData = [];	
		dataCount = 0;	
		dataMean = 0;			
		dataMode = [];	
		longestRow = 0;			
	};

	// 	processLine method - create array to store length value of each row
	Stats.prototype.processLine = function() {
		
		rowLength = [];
	
		for (i = 0; i < data.length; i++) { 		
			var rowDataLength = data[i].length;
			rowLength.push(rowDataLength);						
		}	
	};

	// Calculate and output the required statistics
	Stats.prototype.collectStats = function() {
		
		// Find longest row length in rowLength array
		longestRow = Math.max.apply(null, rowLength);
		
		// Total count of all integers
		dataCount = rowLength.reduce(function(a, b) {
		  return a + b;
		});

		// Combine all data into single arrays
		mergedData = [].concat.apply([], data);;		
		
		// Sum total of integers
		var arraySum = mergedData.reduce(function(a, b) { 
			return a + b;
		});
		
		//Calculate mean
		dataMean = (arraySum / dataCount).toFixed(3);
		
		//Calculate mode from mergedData
		var counter = [];
		var maxValue = 0;
			
		for (i = 0; i < mergedData.length; i++) {			
			// Index count, starting at 0. Increment if value is present
			if (!(mergedData[i] in counter)) {
				counter[mergedData[i]] = 0;
			} else {	
				counter[mergedData[i]]++;
			}			
			// Evaluate if integer has highest count, both independently or shared
			if (counter[mergedData[i]] == maxValue) { 
				dataMode.push(mergedData[i]);
			} else if (counter[mergedData[i]] > maxValue) {
				maxValue = counter[mergedData[i]];
				dataMode = [mergedData[i]];
			}
		}	
	};

	//Update DOM with stat values
	var updateDom = function() {
		function viewModel() {
			this.countOfIntegers = ko.observable(dataCount);
			this.meanOfIntegers = ko.observable(dataMean);
			this.highestLength = ko.observable(longestRow);
			this.commonInteger = ko.observable(dataMode);			
		};
		ko.applyBindings(new viewModel()); 
	};

	//File Upload

	// Confirm browser supports HTML5 File API
	var browserSupportFileUpload = function() {
		var isCompatible = false;
		if(window.File && window.FileReader && window.FileList && window.Blob) {
			isCompatible = true;
		}
		return isCompatible;
	};

	// Upload selected file and create array
	var uploadFile = function(evt) {
		var file = evt.target.files[0];
		var reader = new FileReader();
		reader.readAsText(file);
		reader.onload = function(event) {
			//Jquery.csv
			createArray($.csv.toArrays(event.target.result));			
		};
	};

	// Validate file import
	var createArray = function(data) {	
		if(data !== null && data !== "" && data.length > 1) {
			this.data = data;
			StatsProcessor(data);
			$("#statOutPut").removeClass( "hidden" );			
			$("#errorOutPut").addClass( "hidden" );			
		} else {
			$("#errorOutPut").removeClass( "hidden" );
			$("#statOutPut").addClass( "hidden" );
			$("#errorOutPut li").html('There is no data to import');	
		}	
	};
	
	// event listener for file upload
	if (browserSupportFileUpload()) {
			document.getElementById('txtFileUpload').addEventListener('change', uploadFile, false);
		} else {
			$("#introHeader").html('The File APIs is not fully supported in this browser. Please use another browser.');
		}	
});