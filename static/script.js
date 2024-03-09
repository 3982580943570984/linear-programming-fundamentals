$(document).ready(function () {
  addObjective();
  addEquations();
});

function addObjective() {
  var objective_function = "";

  for (var i = 0; i < $("#variablesCount").val(); ++i) {
    objective_function += `<input type="number" name="c_${i + 1}" placeholder="x${i + 1}" style="margin: 10px; width: 50px">`;
  }

  objective_function += ` = Z`;

  $("#objective").append(objective_function);
}

function addEquation() {
  var equation = '<div class="equation">';

  var i = $("#equations").children().length;
  for (var j = 0; j < $("#variablesCount").val(); ++j) {
    equation += `<input type="number" name="A_${i + 1}_${j + 1}" placeholder="x${j + 1}" style="margin: 10px; width: 50px">`;
  }

  equation += `
    <select name="relation_${i + 1}" style="margin: 10px; width: 50px">
      <option value="<=">&le;</option>
      <option value=">=">&ge;</option>
      <option value="=">=</option>
    </select>

    <input type="number" name="b_${i + 1}" placeholder="b${i + 1}" style="margin: 10px; width: 50px">
    </div>
  `;

  $("#equations").append(equation);
  // updateEquationsCount();
}

const addEquations = () => {
  $("#equations").empty();

  var equationsCount = $("#equationsCount").val();

  for (var i = 0; i < equationsCount; ++i) {
    addEquation();
  }
};

// const removeEquation = (element) => {
//   $(element).closest(".equation").remove();
//   updateEquationsCount();
// };

const updateForm = () => {
  $("#objective").empty();
  $("#equations").empty();

  addObjective();
  addEquations();
};

// function updateEquationsCount() {
//   $("#equationsCount").val($("#equations").children().length);
// }

const convertToCanonical = () => {
  $("#equations")
    .children(".equation")
    .each(function () {
      const relation = $(this).find("select[name^='relation']").val();

      if (relation === "<=") {
        $(this).append(
          `<input type="number" value="1" readonly style="background-color: #e9ecef;">`,
        );
      } else if (relation === ">=") {
        $(this).append(
          `<input type="number" value="1" readonly style="background-color: #e9ecef;">`,
        );
      }

      if (relation !== "=") {
        $(this).find("select[name^='relation']").val("=");
      }
    });
};
