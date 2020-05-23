function process_image(form) {
    var file = document.getElementById('image').value;
    var formData = new FormData(document.querySelector('form'))
    var request = new XMLHttpRequest();
    request.open("POST", "/api/v1/predictor/diagnosis");
    request.send(formData);

    request.onload=function() {
          json=JSON.parse(request.responseText);
          if (json["success"]) {
              var label = json["prediction"]["label"]

              if(label == 0) {
                  document.getElementById("message").innerHTML= "Results are negative!";
              }
              else {
                  document.getElementById("message").innerHTML= "Results are positive!";
              }
          }
          else {
              alert("There was an error processing the request!");
          }
      };

  }

  var dropFileForm = document.getElementById("dropFileForm");
var fileLabelText = document.getElementById("fileLabelText");
var uploadStatus = document.getElementById("uploadStatus");
var fileInput = document.getElementById("fileInput");
var droppedFiles;

function overrideDefault(event) {
  event.preventDefault();
  event.stopPropagation();
}

function fileHover() {
  dropFileForm.classList.add("fileHover");
}

function fileHoverEnd() {
  dropFileForm.classList.remove("fileHover");
}

function addFiles(event) {
  droppedFiles = event.target.files || event.dataTransfer.files;
  showFiles(droppedFiles);
}

function showFiles(files) {
  if (files.length > 1) {
    fileLabelText.innerText = files.length + " files selected";
  } else {
    fileLabelText.innerText = files[0].name;
  }
}

function uploadFiles(event) {
  event.preventDefault();
  changeStatus("Uploading...");

  var formData = new FormData();

  for (var i = 0, file; (file = droppedFiles[i]); i++) {
    formData.append(fileInput.name, file, file.name);
  }

  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function(data) {
    //handle server response and change status of
    //upload process via changeStatus(text)
    console.log(xhr.response);
  };
  xhr.open(dropFileForm.method, dropFileForm.action, true);
  xhr.send(formData);
}

function changeStatus(text) {
  uploadStatus.innerText = text;
}
