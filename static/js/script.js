
	//File Upload

	// event listener for file upload
		

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
		Papa.parse(file, {
			complete: function(results) {
				console.log("AAA: ", results);
				var myJSON = JSON.stringify(results.data);
				console.log("PPPPPP: ", myJSON);


				var myForm = document.getElementById('upload-form')
				var hiddenInput = document.createElement('input')

				hiddenInput.type = 'hidden'
				hiddenInput.name = 'myarray'
				hiddenInput.value = myJSON

				myForm.appendChild(hiddenInput)



			}
		});
	};

if (browserSupportFileUpload()) {
		document.getElementById('txtFileUpload').addEventListener('change', uploadFile, false);
	} else {
		$("#introHeader").html('The File APIs is not fully supported in this browser. Please use another browser.');
	}

// function csvJSON(csv){
//   var lines=csv.split("\n");
//   var result = [];
//   var headers=lines[0].split(";");
//   for(var i=1;i<lines.length;i++){
//       var obj = {};
//       var currentline=lines[i].split(",");
//       for(var j=0;j<headers.length;j++){
//           obj[headers[j]] = currentline[j];
//       }
//       result.push(obj);
//   }
//   return JSON.stringify(result);
// }


