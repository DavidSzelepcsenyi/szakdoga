import sys
import shutil
import math
import cv2
import xml
import xml.etree.cElementTree as ET
import numpy as np
import pytesseract as pytess
from matplotlib import pyplot as plt
import xml.dom.minidom
from PyQt6.QtCore import Qt 
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QDialog, QLineEdit, QCheckBox
from PyQt6.QtGui import QPixmap, QImage

class TextEditDialog(QDialog):
    def __init__(self, roi_image, initial_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Szöveg ellenőrzése")

        # A kapott ROI képet konvertáljuk Qt formátumba
        height, width = roi_image.shape
        bytes_per_line = width
        q_image = QImage(roi_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        
        # 🔹 Egy soros szövegmező
        self.text_edit = QLineEdit(self)
        self.text_edit.setText(initial_text)

        self.image_label = QLabel(self)
        self.image_label.setPixmap(pixmap)
        
        self.underlined_checkbox = QCheckBox("Aláhúzott (Underlined)")
        self.double_checkbox = QCheckBox("Dupla vonalas (Double)")

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Ellenőrzött kép:"))
        layout.addWidget(self.image_label)
        layout.addWidget(QLabel("Felismert szöveg:"))
        layout.addWidget(self.text_edit)
        layout.addWidget(self.underlined_checkbox)
        layout.addWidget(self.double_checkbox)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_text(self):
        return self.text_edit.text()
    
    def get_underlined(self):
        return self.underlined_checkbox.isChecked()

    def get_double(self):
        return self.double_checkbox.isChecked()


class EKtoDrawioApp(QWidget):
    
	avrg_intensity = 0
 
	class Element:
		def __init__(self,id, x, y, width, height, shape, text="", underlined = False, double = False):
			self._id = id
			self._x = x
			self._y = y
			self._width = width
			self._height = height
			self._shape = shape
			self._text = text
			self._underlined = underlined
			self._double = double
			
		def get_id(self):
			return self._id

		def set_id(self, value):
			self._id = value 

		def get_x(self):
			return self._x

		def set_x(self, value):
			self._x = value

		def get_y(self):
			return self._y

		def set_y(self, value):
			self._y = value

		def get_width(self):
			return self._width

		def set_width(self, value):
			self._width = value

		def get_height(self):
			return self._height

		def set_height(self, value):
			self._height = value

		def get_shape(self):
			return self._shape

		def set_shape(self, value):
			self._shape = value

		def get_text(self):
			return self._text

		def set_text(self, value):
			self._text = value

		def get_underlined(self):
			return self._underlined

		def set_underlined(self, value):
			self._underlined = value
   
		def get_double(self):
			return self._double

		def set_double(self, value):
			self._double = value
   
   
	class Line:
		def __init__(self,id,  x1, y1, x2, y2,connection1, connection2, line_type,pointing_at=-1):
			self._id = id
			self._x1 = x1
			self._y1 = y1
			self._x2 = x2
			self._y2 = y2
			self._connection1 = connection1
			self._connection2 = connection2
			self._line_type = line_type
			self._pointing_at = pointing_at

		def get_id(self):
			return self._id

		def set_id(self, value):
			self._id = value 

		def get_x1(self):
			return self._x1

		def set_x1(self, value):
			self._x1 = value

		def get_y1(self):
			return self._y1

		def set_y1(self, value):
			self._y1 = value

		def get_x2(self):
			return self._x2

		def set_x2(self, value):
			self._x2 = value

		def get_y2(self):
			return self._y2

		def set_y2(self, value):
			self._y2 = value

		def get_connection1(self):
			return self._connection1

		def set_connection1(self, value):
			self._connection1 = value

		def get_connection2(self):
			return self._connection2

		def set_connection2(self, value):
			self._connection2 = value

		def get_line_type(self):
			return self._line_type

		def set_line_type(self, value):
			self._line_type = value

		def get_pointing_at(self):
			return self._pointing_at

		def set_pointing_at(self, value):
			self._pointing_at = value
   
	def prepare(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
		avrg_intensity = gray.mean() *0.9


		_, thresh = cv2.threshold(gray, avrg_intensity, 255, cv2.THRESH_BINARY_INV)
	

		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


		cv2.drawContours(img, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
	
		gray_filled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	


		_, binary_filled = cv2.threshold(gray_filled, avrg_intensity, 255, cv2.THRESH_BINARY_INV)

		if avrg_intensity > 210 :
				kernel = np.ones((0, 0), np.uint8)
		else:
			kernel = np.ones((7, 7), np.uint8)  # A kernel méretét a vonalak eltávolításához állítsd be
		opened = cv2.morphologyEx(binary_filled, cv2.MORPH_OPEN, kernel)

		result = cv2.bitwise_not(opened)

		return result, gray

	def fix_mistake(self, im, gray):
		
		mask = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY_INV)[1]
		masked = cv2.bitwise_and(gray, gray, mask=mask)
		contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
		min_area = 200
		max_area = 30000
	
		for contour in contours:
			area = cv2.contourArea(contour)
			if min_area < int(area) < max_area:
				cv2.drawContours(masked, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
				x, y, w, h = cv2.boundingRect(contour)
				max_area = area * 2.6
			else:
				x, y, w, h = cv2.boundingRect(contour)
			
				roi = masked[y:y+h, x:x+w]
	
				avrg_for_inner = gray.mean() *0.97


				_, roi_thresh = cv2.threshold(roi, avrg_for_inner, 255, cv2.THRESH_BINARY_INV)

				inside_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

				for inside_contour in inside_contours:
					inside_area = cv2.contourArea(inside_contour)
					
					if 500 < inside_area < max_area:
						cv2.drawContours(roi, [inside_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
					else:
						cv2.drawContours(roi, [inside_contour], -1, (0, 0, 0), thickness=cv2.FILLED)

				masked[y:y+h, x:x+w] = roi

		return masked

	def angle_between(self, p1, p2, p3):
		a = np.array(p1) - np.array(p2)
		b = np.array(p3) - np.array(p2)
		cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
		angle = np.arccos(cos_theta)
		return np.degrees(angle)

	
	def determine_shape(self, image):
		blurred = cv2.GaussianBlur(image, (5, 5), 0)
		_, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		shapes = []
		id = 0

		for contour in contours:
			if cv2.contourArea(contour) < 500:
				continue


			approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

			contour_points = contour.squeeze()
			side_lengths = []
			for i in range(len(contour_points)):
				p1 = contour_points[i]
				p2 = contour_points[(i + 1) % len(contour_points)]
				side_lengths.append(cv2.norm(p1 - p2))

			angles = []
			for i in range(len(approx)):
				p1 = approx[i - 2][0]
				p2 = approx[i - 1][0]
				p3 = approx[i][0]
				angle = self.angle_between(p1, p2, p3)
				angles.append(angle)

			x, y, w, h = cv2.boundingRect(contour)
			M = cv2.moments(contour)
			if M["m00"] != 0:
				cx = int(M["m10"] / M["m00"])
				cy = int(M["m01"] / M["m00"])

			shape = "Ismeretlen"
			if len(approx) == 3:
				shape = "triangle;whiteSpace=wrap;html=1;"
			elif 5>= len(approx) >= 4:

				if all(85 <= angle <= 95 for angle in angles[:4]):
					if abs(side_lengths[0] - side_lengths[2]) < 10 and abs(side_lengths[1] - side_lengths[3]) < 10:
						shape = "rounded=0;whiteSpace=wrap;html=1;"
					else:
						shape = "rounded=0;whiteSpace=wrap;html=1;"
				elif len(angles) == 4 and abs(angles[0] - angles[2]) < 5 and abs(angles[1] - angles[3]) < 5:
					shape = "rhombus;whiteSpace=wrap;html=1;"
				else:
					shape = "vmi4"
			elif len(approx) > 5:
				shape = "ellipse;whiteSpace=wrap;html=1;"

			elem = self.Element(id, x, y, w, h, shape, "")
			shapes.append(elem)
			id += 1


		return image, shapes

	def remove_shapes(self, image, list, gray):
		
		for element in list:
		
			x = element.get_x()
			y = element.get_y()
			width = element.get_width()
			height = element.get_height()
	
			x1 = max(0, x)
			y1 = max(0, y)
			x2 = min(gray.shape[1], x + width)
			y2 = min(gray.shape[0], y + height)
	
			cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
	
		threshold_value = gray.mean() * 0.8
		_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
	
		return binary_image

	def calculate_angle(self, x1, y1, x2, y2):
     
		return math.degrees(math.atan2(y2 - y1, x2 - x1))

	def merge_lines(self, lines, start, distance_threshold=20, angle_threshold=15):

		merged_lines = []

		for line in lines:
			x1, y1, x2, y2 = line.get_x1(), line.get_y1(), line.get_x2(), line.get_y2()
			angle1 = self.calculate_angle(x1, y1, x2, y2)
			merged = False

			for merged_line in merged_lines:
				mx1, my1, mx2, my2 = merged_line.get_x1(), merged_line.get_y1(), merged_line.get_x2(), merged_line.get_y2()
				angle2 = self.calculate_angle(mx1, my1, mx2, my2)


				dist1 = math.sqrt((x1 - mx2) ** 2 + (y1 - my2) ** 2)
				dist2 = math.sqrt((x2 - mx1) ** 2 + (y2 - my1) ** 2)


				if (dist1 < distance_threshold or dist2 < distance_threshold) and abs(angle1 - angle2) < angle_threshold:

					merged_line.set_x1(min(x1, mx1))
					merged_line.set_y1(min(y1, my1))
					merged_line.set_x2(max(x2, mx2))
					merged_line.set_y2(max(y2, my2))
					merged = True
					break

			if not merged:
				merged_lines.append(line)
				
		for line in merged_lines:
			line.set_id(start)
			start += 1
		
		return merged_lines


	def find_lines(self, image, shapes):

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		

		adaptive_thresh = cv2.adaptiveThreshold(
			blurred, 255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY_INV,
			11, 2 
		)


		kernel = np.ones((3, 3), np.uint8)
		adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
		adaptive_thresh = cv2.dilate(adaptive_thresh, kernel, iterations=1)
		adaptive_thresh = cv2.erode(adaptive_thresh, kernel, iterations=1)


		contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		min_contour_area = 20
		for contour in contours:
			if cv2.contourArea(contour) < min_contour_area:
				cv2.drawContours(adaptive_thresh, [contour], -1, 0, thickness=cv2.FILLED)



		lines = cv2.HoughLinesP(
			adaptive_thresh, 
			rho=1, 
			theta=np.pi / 180, 
			threshold=10, 
			minLineLength=5,
			maxLineGap=5
		)

		lines_data = []
		id = len(shapes)
	
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				a_line = self.Line(id, x1, y1, x2, y2, 0, 0, "Line")
				lines_data.append(a_line)
				id += 1

		merged_lines = self.merge_lines(lines_data, len(shapes), 5,15)

		return merged_lines

	def associate_lines_to_shapes(self,lines_data, shapes_list):

		line_coords = np.array([[line.get_x1(), line.get_y1(), line.get_x2(), line.get_y2()] for line in lines_data])

		for line, (x1, y1, x2, y2) in zip(lines_data, line_coords):
			closest_id1 = -1
			closest_dist1 = 75
			closest_id2 = -1
			closest_dist2 = 75

			for elem in shapes_list:
				eID, ex, ey, ew, eh = elem.get_id(), elem.get_x(), elem.get_y(), elem.get_width(), elem.get_height()


				x_range = np.arange(ex, ex + ew)
				y_range = np.arange(ey, ey + eh)
				x_grid, y_grid = np.meshgrid(x_range, y_range)


				points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)


				dist1 = np.sqrt((points[:, 0] - x1)**2 + (points[:, 1] - y1)**2)
				dist2 = np.sqrt((points[:, 0] - x2)**2 + (points[:, 1] - y2)**2)


				min_dist1 = np.min(dist1)
				min_dist2 = np.min(dist2)

				if min_dist1 < closest_dist1:
					closest_dist1 = min_dist1
					closest_id1 = eID

				if min_dist2 < closest_dist2:
					closest_dist2 = min_dist2
					closest_id2 = eID


			line.set_connection1(closest_id1)
			line.set_connection2(closest_id2)

		return lines_data

	def validate_lines(self,lines, shapes):
		valid_lines = []
		true_valid_lines = []  
		
		for line in lines:
			valid = True
			c1, c2 = line.get_connection1(), line.get_connection2()
			
			if c1 == c2:
				valid = False
				continue
			
			if c1 >= 0 and c2 >= 0 and shapes[c1].get_shape() == shapes[c2].get_shape():
				shape = shapes[c1].get_shape()
				if shape not in ["ellipse;whiteSpace=wrap;html=1;", "vmi4", "Ismeretlen"]:
					valid = False
     
			if shapes[c1].get_shape() == "rhombus;whiteSpace=wrap;html=1;" and shapes[c2].get_shape() == "ellipse;whiteSpace=wrap;html=1;":
					valid = False
				
			if shapes[c2].get_shape() == "rhombus;whiteSpace=wrap;html=1;" and shapes[c1].get_shape() == "ellipse;whiteSpace=wrap;html=1;":
				valid = False

			for vonal in valid_lines:
				con1, con2 = vonal.get_connection1(), vonal.get_connection2()

				if (c1 == con1 and c2 == con2) or (c1 == con2 and c2 == con1):

					current_length = ((line.get_x2() - line.get_x1()) ** 2 + (line.get_y2() - line.get_y1()) ** 2) ** 0.5
					existing_length = ((vonal.get_x2() - vonal.get_x1()) ** 2 + (vonal.get_y2() - vonal.get_y1()) ** 2) ** 0.5

					if current_length > existing_length:
						valid_lines.remove(vonal)
						valid = True
					else:
						valid = False

			if valid:
				valid_lines.append(line)

		for line in valid_lines:
			weak1, weak2 = True, True
			c1, c2 = line.get_connection1(), line.get_connection2()
			
			if c1 >= 0 and c2 >= 0 and shapes[c1].get_shape() == "ellipse;whiteSpace=wrap;html=1;" and shapes[c2].get_shape() == "ellipse;whiteSpace=wrap;html=1;":
				for vonal in valid_lines:
					if vonal.get_connection1() == c1 or vonal.get_connection2() == c1:
						if vonal.get_connection1() >= 0 and vonal.get_connection2() >= 0:
							if shapes[vonal.get_connection1()].get_shape() != "ellipse;whiteSpace=wrap;html=1;" or \
							shapes[vonal.get_connection2()].get_shape() != "ellipse;whiteSpace=wrap;html=1;":
								weak1 = False
					if vonal.get_connection1() == c2 or vonal.get_connection2() == c2:
						if vonal.get_connection1() >= 0 and vonal.get_connection2() >= 0:
							if shapes[vonal.get_connection1()].get_shape() != "ellipse;whiteSpace=wrap;html=1;" or \
							shapes[vonal.get_connection2()].get_shape() != "ellipse;whiteSpace=wrap;html=1;":
								weak2 = False
			
			if weak1 or weak2:
				true_valid_lines.append(line)

		return true_valid_lines


	def complex_line_checker(self, valid_lines, lines, shapes):
		for i in range(0,len(valid_lines)):
			if valid_lines[i].get_connection1() == -1 or valid_lines[i].get_connection2() == -1:
				fx1, fy1 = -1, -1

				if valid_lines[i].get_connection1() == -1:
					fx1 = valid_lines[i].get_x1()
					fy1 = valid_lines[i].get_y1()

				if valid_lines[i].get_connection2() == -1:
					fx1 = valid_lines[i].get_x2()
					fy1 = valid_lines[i].get_y2()

				for j in range(i+1,len(valid_lines)):
					fx2,fy2 = -1, -1
					if valid_lines[j].get_connection1() == -1:
						fx2 = valid_lines[j].get_x1()
						fy2 = valid_lines[j].get_y1()
		
					if valid_lines[j].get_connection2() == -1:
						fx2 = valid_lines[j].get_x2()
						fy2 = valid_lines[j].get_y2()

					if fx1 >= 0 and fy1 >= 0 and fx2 >= 0 and fy2 >= 0:


						min_x = min(fx1, fx2) - 20
						max_x = max(fx1, fx2) + 20
						min_y = min(fy1, fy2) - 20
						max_y = max(fy1, fy2) + 20
		
						overlap = False
						for shape in shapes:
							sx, sy, sw, sh = shape.get_x(), shape.get_y(), shape.get_width(), shape.get_height()

							if not (max_x < sx or min_x > sx + sw or max_y < sy or min_y > sy + sh):
								overlap = True
								break
	
						if overlap:
							continue

						candidate_lines = []
						for line in lines:
							lx1, ly1, lx2, ly2 = line.get_x1(), line.get_y1(), line.get_x2(), line.get_y2()

							if min_x <= lx1 <= max_x and min_y <= ly1 <= max_y and \
								min_x <= lx2 <= max_x and min_y <= ly2 <= max_y:
								candidate_lines.append(line)


						if len(candidate_lines) >= 1:

							closest_x1, closest_y1 = candidate_lines[0].get_x1(), candidate_lines[0].get_y1()
							closest_x2, closest_y2 = candidate_lines[0].get_x2(), candidate_lines[0].get_y2()
							min_dist1 = math.sqrt((fx1 - closest_x1) ** 2 + (fy1 - closest_y1) ** 2)
							min_dist2 = math.sqrt((fx2 - closest_x2) ** 2 + (fy2 - closest_y2) ** 2)

							for line in candidate_lines:
								for x, y in [(line.get_x1(), line.get_y1()), (line.get_x2(), line.get_y2())]:
									dist1 = math.sqrt((fx1 - x) ** 2 + (fy1 - y) ** 2)
									dist2 = math.sqrt((fx2 - x) ** 2 + (fy2 - y) ** 2)
									
									if dist1 < min_dist1:
										closest_x1, closest_y1 = x, y
										min_dist1 = dist1

									if dist2 < min_dist2:
										closest_x2, closest_y2 = x, y
										min_dist2 = dist2
							new_line = self.Line(
								id= lines[len(lines) -1].get_id() + 1,
								x1=closest_x1, 
								y1=closest_y1, 
								x2=closest_x2, 
								y2=closest_y2, 
								connection1=valid_lines[i].get_id(), 
								connection2=valid_lines[j].get_id(),
								line_type= "Connector"
							)

							valid_lines.append(new_line)
							lines.append(new_line)
							to_modify1 = valid_lines[i].get_id()
							to_modify2 = valid_lines[j].get_id()
		
							for line in valid_lines:
								if line.get_id() == to_modify1:
									if line.get_connection1() == -1:
										line.set_connection1(new_line.get_id())
									elif line.get_connection2() == -1:
										line.set_connection2(new_line.get_id())
								if line.get_id() == to_modify2:
									if line.get_connection1() == -1:
										line.set_connection1(new_line.get_id())
									elif line.get_connection2() == -1:
										line.set_connection2(new_line.get_id())
							for line in lines:
								if line.get_id() == to_modify1:
									if line.get_connection1() == -1:
										line.set_connection1(new_line.get_id())
									elif line.get_connection2() == -1:
										line.set_connection2(new_line.get_id())
								if line.get_id() == to_modify2:
									if line.get_connection1() == -1:
										line.set_connection1(new_line.get_id())
									elif line.get_connection2() == -1:
										line.set_connection2(new_line.get_id())

			
	def follow_connectors(self, original,start_index, lines, shapes):
		used = set()
		current = start_index
		end_x, end_y = None, None
		used.add(original)
		line = None

		while current >= len(shapes):
			used.add(current)
			
			line = lines[current - len(shapes)]
			if line.get_connection1() < len(shapes):
				current = line.get_connection1()
				end_x, end_y = line.get_x1(), line.get_y1()
				break
			elif line.get_connection2() < len(shapes):
				current = line.get_connection2()
				end_x, end_y = line.get_x2(), line.get_y2()
				break
			else:
				if line.get_connection1() not in used:
					current = line.get_connection1()
					end_x, end_y = line.get_x1(), line.get_y1()
				else:
					current = line.get_connection2()
					end_x, end_y = line.get_x2(), line.get_y2()

		return current, end_x, end_y, line.get_id()

	def closest_border_distance(self, checking_point_x, checking_point_y, shape):

		sx, sy, sw, sh = shape.get_x(), shape.get_y(), shape.get_width(), shape.get_height()

		border_points = []

		for x in range(sx, sx + sw + 1):
			border_points.append((x, sy))
			border_points.append((x, sy + sh))

		for y in range(sy, sy + sh + 1):
			border_points.append((sx, y))
			border_points.append((sx + sw, y))


		min_distance = float("inf")
		closest_point = None

		for bx, by in border_points:
			dist = math.sqrt((checking_point_x - bx) ** 2 + (checking_point_y - by) ** 2)
			if dist < min_distance:
				min_distance = dist
				closest_point = (bx, by)

		return min_distance, closest_point

	def count_lines_near_midpoint(self, checking_point_x, checking_point_y, closest_point_x, closest_point_y, min_distance, lines, valid_lines):

		mid_x = (checking_point_x + closest_point_x) / 2
		mid_y = (checking_point_y + closest_point_y) / 2
		search_radius = min_distance-2 / 2

		possible_arrow_parts = []
		
		for line in lines:
		
			lx1, ly1, lx2, ly2 = line.get_x1(), line.get_y1(), line.get_x2(), line.get_y2()

			dist1 = math.sqrt((mid_x - lx1) ** 2 + (mid_y - ly1) ** 2)
			dist2 = math.sqrt((mid_x - lx2) ** 2 + (mid_y - ly2) ** 2)

			if dist1 <= search_radius or dist2 <= search_radius:
				possible_arrow_parts.append(line)

		return possible_arrow_parts



	def arrow_checker(self, valid_lines, lines, shapes):
		for v in valid_lines:
			if v.get_line_type() == "Line":
				origin = v.get_id()
				c1, c2 = v.get_connection1(), v.get_connection2()
				checking_point_x, checking_point_y = None, None
				checking_line = v.get_id()
				checking_shape = None
				connected_line = None
				c1_belongs_to_original = True
				c2_belongs_to_original = True

				if c1 >= len(shapes):
					c1, end_x1, end_y1, connected_line = self.follow_connectors(origin,c1, lines, shapes)
					c1_belongs_to_original = False
				else:
					end_x1, end_y1 = v.get_x1(), v.get_y1()
				if c2 >= len(shapes):
					c2, end_x2, end_y2, connected_line = self.follow_connectors(origin,c2, lines, shapes)
					c2_belongs_to_original = False
				else:
					end_x2, end_y2 = v.get_x2(), v.get_y2()

				if c1 < len(shapes) and shapes[c1].get_shape() == "rhombus;whiteSpace=wrap;html=1;":
					checking_point_x, checking_point_y = end_x2, end_y2
					checking_shape = c2
					if not c2_belongs_to_original:
						checking_line = connected_line
				if c2 < len(shapes) and shapes[c2].get_shape() == "rhombus;whiteSpace=wrap;html=1;":
					checking_point_x, checking_point_y = end_x1, end_y1
					checking_shape = c1
					if not c1_belongs_to_original:
						checking_line = connected_line

				if checking_point_x is not None and checking_point_y is not None:
					search_radius = 6
					found_lines = []
					for line in lines :
						if line not in valid_lines and (
							(checking_point_x - search_radius <= line.get_x1() <= checking_point_x + search_radius and
								checking_point_y - search_radius <= line.get_y1() <= checking_point_y + search_radius) or
							(checking_point_x - search_radius <= line.get_x2() <= checking_point_x + search_radius and
								checking_point_y - search_radius <= line.get_y2() <= checking_point_y + search_radius)
						):
							found_lines.append(line)

					
					main_line = lines[checking_line-len(shapes)]
		
					min_distance, closest_point = self.closest_border_distance(checking_point_x, checking_point_y, shapes[checking_shape])
		
					main_angle = self.calculate_angle(main_line.get_x1(), main_line.get_y1(), main_line.get_x2(), main_line.get_y2())
					arrow_parts = []
					if min_distance < 5:

						for fl in found_lines:
							fl_angle = self.calculate_angle(fl.get_x1(), fl.get_y1(), fl.get_x2(), fl.get_y2())
							fl_angle = (fl_angle + 360) % 360
							angle_diff = abs(main_angle - fl_angle)

							if 25 <= angle_diff <= 75:
									arrow_parts.append(fl)

						if len(arrow_parts) >= 2:
							v.set_line_type("Arrow")
							v.set_pointing_at(checking_shape)
					else:
						midpoint_lines = self.count_lines_near_midpoint(
							checking_point_x, checking_point_y,
							closest_point[0], closest_point[1],
							min_distance, lines,valid_lines
						)

						for mp_line in midpoint_lines:
							mp_angle = self.calculate_angle(mp_line.get_x1(), mp_line.get_y1(), mp_line.get_x2(), mp_line.get_y2())
							mp_angle = (mp_angle + 360) % 360
							angle_diff = abs(main_angle - mp_angle)



							if 25 <= angle_diff <= 75:
								arrow_parts.append(mp_line)

						if len(arrow_parts) >= 2 and checking_line == v.get_id():
							v.set_line_type("Arrow")
							v.set_pointing_at(checking_shape)
							



	def check_shapes(self,shapes,lines):
		for shape in shapes:
			evidence = []
			para = 0
			negyzet = 0
			if shape.get_shape() == "Ismeretlen" or shape.get_shape() == "vmi4":
				for line in lines:
					if shape.get_id() == line.get_connection1() and line.get_connection2() >=0 :
						evidence.append(shapes[line.get_connection2()])
					if shape.get_id() == line.get_connection2() and line.get_connection1() >= 0:
						evidence.append(shapes[line.get_connection1()])
				for item in evidence:
					if item.get_shape() in ["triangle;whiteSpace=wrap;html=1;", "rhombus;whiteSpace=wrap;html=1;", "ellipse;whiteSpace=wrap;html=1;"]:
						negyzet += 1
					else:
						para +=1
				if para > negyzet:
					shape.set_shape("rhombus;whiteSpace=wrap;html=1;")
				elif para < negyzet:
					shape.set_shape("rounded=0;whiteSpace=wrap;html=1;")
				else:
					shape.set_shape("Ismeretlen")
					
	def make_XML(self,shapes, lines,all_lines):
		mxfile = ET.Element(
			"mxfile",
			host="app.diagrams.net",
			agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 OPR/114.0.0.0",
			version="26.0.14"
		)
		diagram = ET.SubElement(mxfile, "diagram", name="Page-1", id="E5nG0SyLeEiv9CRzUDmB")

		mxGraphModel = ET.SubElement(
			diagram,
			"mxGraphModel",
			dx="2033",
			dy="1123",
			grid="1",
			gridSize="10",
			guides="1",
			tooltips="1",
			connect="1",
			arrows="1",
			fold="1",
			page="1",
			pageScale="1",
			pageWidth="900",
			pageHeight="1200",
			math="0",
			shadow="0"
		)

		root = ET.SubElement(mxGraphModel, "root")

		ET.SubElement(root, "mxCell", id="0")
		ET.SubElement(root, "mxCell", id="1", parent="0")

		for elem in shapes:
			cell_attrib = {
				"id": str(elem.get_id() + 2),
				"value": elem.get_text(),
				"style": elem.get_shape(),
				"vertex": "1",
				"parent": "1"	
			}
			if elem.get_underlined():
				cell_attrib["style"] += "fontStyle=4;"
    
			if elem.get_double():
				if elem.get_shape() == "rounded=0;whiteSpace=wrap;html=1;" :
					cell_attrib["style"] = "shape=ext;margin=3;double=1;whiteSpace=wrap;html=1;align=center;"
				elif elem.get_shape() == "rhombus;whiteSpace=wrap;html=1;" :
					cell_attrib["style"] = "shape=rhombus;double=1;perimeter=rhombusPerimeter;whiteSpace=wrap;html=1;align=center;"
				elif elem.get_shape() == "ellipse;whiteSpace=wrap;html=1;" :
					cell_attrib["style"] = "ellipse;shape=doubleEllipse;margin=3;whiteSpace=wrap;html=1;align=center;"
       
			mxCell = ET.SubElement(root,"mxCell", attrib=cell_attrib)
			ET.SubElement(
				mxCell, 
				"mxGeometry", 
				attrib={
					"x": str(elem.get_x()),
					"y": str(elem.get_y()),
					"width": str(elem.get_width()),
					"height": str(elem.get_height()),
					"as": "geometry"
				}
			)

		for line in lines:
			cell_attrib = {
				"id": str(line.get_id() + 2),
				"value": "",
				"style": "endArrow=none;html=1;rounded=0;",
				"edge": "1",
				"parent": "1",
				"source": str(line.get_connection1() + 2),
				"target": str(line.get_connection2() + 2)
			}

			p1_attrib = {
				"x": str(line.get_x1()),
				"y": str(line.get_y1()),
				"as": "sourcePoint"
			}
			p2_attrib = {
				"x": str(line.get_x2()),
				"y": str(line.get_y2()),
				"as": "targetPoint"
			}

			if line.get_line_type() == "Arrow":
				cell_attrib["style"] = "endArrow=classic;html=1;rounded=0;"
				
				
			if line.get_pointing_at() > -1:
				if line.get_connection1() == line.get_pointing_at():
					cell_attrib["source"], cell_attrib["target"] = cell_attrib["target"], cell_attrib["source"]
					p1_attrib["x"], p1_attrib["y"] = str(line.get_x2()), str(line.get_y2())
					p2_attrib["x"], p1_attrib["y"] = str(line.get_x1()), str(line.get_y1())
		

			if int(cell_attrib["source"]) >= len(shapes) + 2:
				c = all_lines[int(cell_attrib["source"]) - 2 - len(shapes)]
				if c.get_connection1() == line.get_id():   
					p1_attrib["x"], p1_attrib["y"] = str(c.get_x1()), str(c.get_y1())
				elif c.get_connection2() == line.get_id():   
					p1_attrib["x"], p1_attrib["y"] = str(c.get_x2()), str(c.get_y2())
				cell_attrib["source"] = ""
			elif int(cell_attrib["target"]) >= len(shapes) + 2:
				c = all_lines[int(cell_attrib["target"]) - 2 - len(shapes)]
				if c.get_connection1() == line.get_id():   
					p1_attrib["x"], p1_attrib["y"] = str(c.get_x1()), str(c.get_y1())
				elif c.get_connection2() == line.get_id():   
					p1_attrib["x"], p1_attrib["y"] = str(c.get_x2()), str(c.get_y2())
				cell_attrib["target"] = ""
				
			if line.get_line_type() == "Connector":
				cell_attrib["source"], cell_attrib["target"] = "", ""

			mxCell = ET.SubElement(root, "mxCell", attrib=cell_attrib)

			mxGeometry = ET.SubElement(
				mxCell, 
				"mxGeometry", 
				attrib={
					"width": "50",
					"height": "50",
					"relative": "1",
					"as": "geometry"
				}
			)
			ET.SubElement(mxGeometry, "mxPoint", attrib=p1_attrib)
			ET.SubElement(mxGeometry, "mxPoint", attrib=p2_attrib)


		xml_str = ET.tostring(mxfile, encoding="utf-8")
		pretty_xml = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="    ")

		with open("result.drawio", "w", encoding="utf-8") as f:
			f.write(pretty_xml)


	def find_text(self,shapes_list,gray):
		for elem in shapes_list:
			x, y, width, height = elem.get_x(), elem.get_y(), elem.get_width(), elem.get_height()

			x1 = max(0, x + 2)
			y1 = max(0, y + 2)
			x2 = min(gray.shape[1], x + width - 2)
			y2 = min(gray.shape[0], y + height - 2)

			roi = gray[y1:y2, x1:x2]
			roi = cv2.resize(roi, (0, 0), fx=2.0, fy=2.0)

			sharpening_kernel = np.array([[0, -1, 0],
										[-1, 5, -1],
										[0, -1, 0]])


			sharp = cv2.filter2D(roi, -1, sharpening_kernel)

			text = pytess.image_to_string(sharp)
			clean_text = text.replace("\n", " ").replace("\r", " ")
			dialog = TextEditDialog(roi, clean_text)
			result = dialog.exec()  # Ez blokkolja a futást, amíg a felhasználó nem nyom OK-t

			if result == QDialog.DialogCode.Accepted:
				clean_text = dialog.get_text()  # Az új szöveg betöltése
				underlined = dialog.get_underlined()
				double = dialog.get_double()
    
			elem.set_text(clean_text)
			elem.set_underlined(underlined)
			elem.set_double(double)

	def __init__(self):
		super().__init__()


		self.setWindowTitle("E-K to Drawio")
		self.setFixedSize(600, 500)
		self.uploaded_file_path = None  # Feltöltött fájl nyomon követése

		# Fő elrendezés
		layout = QVBoxLayout()
		layout.setAlignment(Qt.AlignmentFlag.AlignTop)

		# Cím középen
		title_label = QLabel("Szabályok")
		title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
		title_label.setStyleSheet("font-size: 24px; font-weight: bold;")

		# Bal oldali szöveg (szabályok)
		left_text_label = QLabel(
			"A program rendes működése csak bizonyos feltételek mellett biztosított."
			"<br> 1. A diagram csak az EK diagram szokásos elemeit tartalmazza."
			"<br> 2. A diagram fekete tollal, fehér lapra legyen rajzolva, és csak a diagram legyen a lapon."
			"<br> 3. A lap másik oldalról ne üssön át más."
			"<br> 4. A vonalak minél pontosabbak és egyenesebbek legyenek, vonalzó használata javasolt."
			"<br> 5. Ha egy vonalat vissza akarunk vezetni az eredeti testre, elsőnek távolodjon el a testtől, majd húzzon egyenes vonalat,"
			"<br> ha kell több lépésben."
			"<br> 6. A vonalak legyenek kellően hosszúak, és ne legyen bennük törés, szakadékosság."
			"<br> 7. Ne érjenek a tesztek és vonalak egybe, legyen kellő távolság köztük."
		)
		left_text_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
		left_text_label.setStyleSheet("font-size: 14px;")

		# Fájlfeltöltő gomb
		upload_button = QPushButton("Fájl feltöltése")
		upload_button.clicked.connect(self.upload_file)

		# Feltöltött fájl nevét mutató címke
		self.file_label = QLabel("Nincs feltöltött fájl")
		self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
		
		# Konvertálás gomb
		convert_button = QPushButton("Konvertálás")
		convert_button.clicked.connect(self.process_file)

		# Elemek hozzáadása az elrendezéshez
		layout.addWidget(title_label)  # Cím
		layout.addWidget(left_text_label)  # Bal oldali szöveg (szabályok)
		layout.addWidget(upload_button)  # Fájlfeltöltő gomb
		layout.addWidget(self.file_label)  # Feltöltött fájl neve
		layout.addWidget(convert_button)  # Konvertálás gomb

		# Beállítjuk az elrendezést az ablakra
		self.setLayout(layout)

	def upload_file(self):
		file_dialog = QFileDialog()
		file_path, _ = file_dialog.getOpenFileName(self, "Válassz egy fájlt")

		if file_path:  # Ha a felhasználó választott fájlt
			self.uploaded_file_path = file_path
			shutil.copy(file_path, "img")  # Másolás "img" néven
			self.file_label.setText(f"Feltöltött fájl: {file_path.split('/')[-1]}")  # Megjeleníti a fájl nevét

	def process_file(self):
		if not self.uploaded_file_path:
			QMessageBox.warning(self, "Hiba", "Nincs feltöltött fájl!")
			return

		# 🔹 1. Kép beolvasása
		img_o = cv2.imread("img")
		img = cv2.resize(img_o, (770, 512), fx=1.0, fy=1.0)
		copy = cv2.resize(img_o, (770, 512), fx=1.0, fy=1.0)

		# 🔹 2. Fő lépések végrehajtása
		prepared, gray = self.prepare(img)
		res = self.fix_mistake(prepared, gray)
		res2, shapes_list = self.determine_shape(res)
		lines = self.remove_shapes(copy, shapes_list,gray)
		lines_data = self.find_lines(lines, shapes_list)
		self.find_text(shapes_list,gray)
		lines_data = self.associate_lines_to_shapes(lines_data, shapes_list)
		valid_lines = self.validate_lines(lines_data, shapes_list)
		self.check_shapes(shapes_list, valid_lines)
		self.check_shapes(shapes_list, valid_lines)
		self.complex_line_checker(valid_lines, lines_data, shapes_list)
		self.arrow_checker(valid_lines, lines_data, shapes_list)


		# 🔹 3. XML generálás
		self.make_XML(shapes_list, valid_lines, lines_data)

		# 🔹 4. Sikerüzenet
		QMessageBox.information(self, "Siker", "A fájl feldolgozása befejeződött!\nLétrejött a result.drawio fájl.")
  


  
# Alkalmazás indítása
app = QApplication(sys.argv)
window = EKtoDrawioApp()
window.show()
sys.exit(app.exec())