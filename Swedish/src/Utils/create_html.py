import os
import shutil
import csv
def create_html(wrong_pred, real_img, save_dir):
    table_row_str = """
    <tr>
        <td><img src="{}" width="200"></td>
        <td><h1>{}</h1></td>
        <td><h1>{}</h1></td>
    </tr>
"""
    with open(f"{save_dir}/visual.html", "w", newline='') as f:
        f.write("""
<style>
    table, th, td {
        border: 1px solid;
    }
</style>
""")
        f.write("<table>\n")
        f.write("""
    <tr>
        <th><h1>Image<h1></th>
        <th><h1>Prediction</h1></th>
        <th><h1>Label</h1></th>
    </tr>
""")
        for pred in wrong_pred.keys():
            f.write(table_row_str.format(os.path.join("..", real_img[pred]), pred, ""))
            for label in wrong_pred[pred].keys():
                for img in wrong_pred[pred][label]:
                    f.write(table_row_str.format(os.path.join("..", img), pred, label))
            f.write('<tr bgcolor="black" style="height:200;" ><td></td><td></td><td></td></tr>')
        
        f.write("</table>")
