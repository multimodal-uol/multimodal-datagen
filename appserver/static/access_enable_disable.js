require([
     "splunkjs/mvc",
     "splunkjs/mvc/simplexml/ready!"
 ], function(
     mvc
 ) {
     console.log("Inside Enable Disable Access JS");
     var defaultTokenModel = mvc.Components.get("default");
     defaultTokenModel.on("change:access", function(newValue) {
         var tokAccess = defaultTokenModel.get("access");
         if (tokAccess === "disabled") {
             // Disable TextBox Input with id="text1"
             $("#text1 .splunk-textinput input[type='text']").attr("disabled", "disabled");
             // Disable Time Picker Input with id="time1"
             $("#time1 .splunk-timerange button[type='button']").attr("disabled", "disabled");
             // Disable Dropdown Input with id="dropdown1"
             $("#dropdown1 .splunk-dropdown button[type='button']").attr("disabled", "disabled");
 
         } else {
             // Enable TextBox Input with id="text1"
             $(".splunk-textinput input[type='text']").removeAttr("disabled");
             // Enable Time Picker Input with id="time1"
             $("#time1 .splunk-timerange button[type='button']").removeAttr("disabled");
             // Enable Dropdown Input with id="dropdown1"
             $("#dropdown1 .splunk-dropdown button[type='button']").removeAttr("disabled");
         }
     });
 
 
     $(document).on("click", "#time1 div.splunk-timerange div[data-test='time-range-dropdown'] span[data-test='label']", function() {
         var strPopOverIDTime = "div#" +$("#time1 div.splunk-timerange div[data-test='time-range-dropdown']").attr("data-test-popover-id");
         console.log("Popover ID:", strPopOverIDTime);
         setTimeout(function() {
             $(strPopOverIDTime).hide();
         }, 10);
     });
     $(document).on("click", "#dropdown1 div.splunk-dropdown div[data-test='select']", function() {
         var strPopOverID = "div#" + $(this).attr("data-test-popover-id");
         console.log("Popover ID:", strPopOverID);
         setTimeout(function() {
             $(strPopOverID).hide();
         }, 10);
     });
 });

