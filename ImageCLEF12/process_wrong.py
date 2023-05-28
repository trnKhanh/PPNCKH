import os
import shutil
import csv
if __name__ == '__main__':
    root_dir = './5_mobilev2_scan'
    table_row_str = """
    <tr>
        <td><img src="{}" width="200"></td>
        <td><h1>{}</h1></td>
        <td><h1>{}</h1></td>
    </tr>
"""
    with open(f"{root_dir}_pred.html", "w", newline='') as f:
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
        for pred in os.scandir(root_dir):
            if not os.path.isdir(pred.path):
                continue
            for label in os.scandir(pred.path):
                if not os.path.isdir(label.path):
                    f.write(table_row_str.format(label.path, pred.name.split("_")[1], ""))
                    break

            for label in os.scandir(pred.path):
                if not os.path.isdir(label.path):
                    continue
                for img in os.scandir(label.path):
                    f.write(table_row_str.format(img.path, pred.name.split("_")[1], label.name))
            f.write('<tr bgcolor="black" style="height:200;" ><td></td><td></td><td></td></tr>')
        
        f.write("</table>")
