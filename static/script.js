$(document).ready(function () {
  addObjective();
  addEquations();
});

const addObjective = () => {
  let objective_function = "";

  for (let i = 0; i < $("#variables_count").val(); ++i) {
    objective_function += `<input type="number" name="c_${i + 1}" placeholder="x${i + 1}" style="margin: 10px; width: 50px">`;
  }

  objective_function += ` = Z`;

  $("#objective").append(objective_function);
};

const addEquation = () => {
  let equation = '<div class="equation">';

  const i = $("#equations").children().length;
  for (let j = 0; j < $("#variables_count").val(); ++j) {
    equation += `<input type="number" name="A_${i + 1}_${j + 1}" placeholder="x${j + 1}" style="margin: 10px; width: 50px">`;
  }

  equation += `
    <select name="sign_${i + 1}" style="margin: 10px; width: 50px">
      <option value="<=">&le;</option>
      <option value=">=">&ge;</option>
      <option value="=">=</option>
    </select>

    <input type="number" name="b_${i + 1}" placeholder="b${i + 1}" style="margin: 10px; width: 50px">
    </div>
  `;

  $("#equations").append(equation);
};

const addEquations = () => {
  $("#equations").empty();

  const equations_count = $("#equations_count").val();
  for (let i = 0; i < equations_count; ++i) {
    addEquation();
  }
};

const updateForm = () => {
  $("#objective").empty();
  $("#equations").empty();

  addObjective();
  addEquations();
};
